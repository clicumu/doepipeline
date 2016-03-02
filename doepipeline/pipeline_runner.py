import paramiko
import time
import posixpath
import yaml
import sys
import logging
import re
from pipeline_writer import write_pipeline

# Get the logger
log = logging.getLogger(__name__)


class PipelineRunner(object):
    """
    Class specific for DoE Optimization project by Daniel Svensson.
    Starts at iteration 0.
    """

    def __init__(self, pipeline_YAML, stop_time=3600, connect_wait=10, attempts=60, check_interval=20, iteration = 0):
        self.stop_time = stop_time
        self.connect_wait = connect_wait
        self.attempts = attempts
        self.check_interval = check_interval
        self.client = None
        self.pid_dict = {}
        self.exp_design = None
        self.base_logfile = "stdout_stderr.log"
        self.yaml_dict = self.yaml_to_dict(pipeline_YAML)
        self.iteration = iteration
        self.pipeline = None
        self.num_exp = None

    def yaml_to_dict(self, yaml_settings_file):
        with open(yaml_settings_file, 'r') as settings:
            settings_dict = yaml.load(settings)
        return settings_dict

    def new_pipeline(self):
        """
        Write a new pipeline of experiments to run (first run set_exp_design)
        """
        if self.exp_design is not None:
            self.pipeline = write_pipeline(
                experimental_setup=self.exp_design,
                yaml_dict=self.yaml_dict,
                iteration=self.iteration
            )
            self.num_exp = len(self.pipeline)
        else:
            sys.exit("Can't find an experimental design. First, supply an experimental design with the function 'set_exp_design'. Exiting...")

    def get_pipeline(self):
        if self.pipeline:
            return self.pipeline
        else:
            sys.exit('No pipeline exists. First, create one using the function "new_pipeline(exp_design)". Exiting...')

    def paramiko_connect(self):
        """
        Connect to the server using paramiko.
        """
        attempts = 0
        while True:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.load_system_host_keys()
                client.connect(
                    self.yaml_dict['host'],
                    username=self.yaml_dict['user'],
                    password=self.yaml_dict['password'],
                    key_filename=self.yaml_dict['private_key']
                )
            except paramiko:
                print "Cannot connect, retrying..."
                attempts += 1
                time.sleep(self.connect_wait)
            else:
                print "connected!"
                return client

            if attempts == 60:
                return False

        return True

    def paramiko_disconnect(self):
        print "disconnecting..."
        self.client.close()
        self.client = None

    def set_exp_design(self, exp_design):
        self.exp_design = exp_design

    def make_directories(self):
        # self.client != None when running this.
        print "creating directories in %s..." % self.yaml_dict['kaw_workdir']
        for exp_num in range(1, self.num_exp + 1):
            output_dir = posixpath.join(
                self.yaml_dict['kaw_workdir'],
                self.yaml_dict['inv_name'],
                str(self.iteration),
                str(exp_num)
            )
            print "Creating directory %s" % output_dir
            self.client.exec_command('mkdir -p %s' % output_dir)

    def setup_screens(self):
        print "setting up screens..."
        for exp_num in range(1, self.num_exp + 1):
            self.client.exec_command('screen -S %s' % str(exp_num))  # Set up a screen for this experiment
            self.client.exec_command('screen -d')  # Detach from the screen

    def initiate_experiments(self):
        self.setup_screens()
        for run_order in range(1, self.num_exp+1):
            experiment_settings = self.exp_design.loc[self.exp_design['Run Order'] == run_order]
            exp_num = str(experiment_settings.iloc[0]['Exp No'])
            pid = self.run_command(exp_num, pipeline_step=1)
            self.pid_dict[exp_num] = {
                "pid": pid,
                "subprocess": 1,
            }

    def run_command(self, exp_num, pipeline_step):
        logfile = str(pipeline_step) + "_" + self.base_logfile
        abs_logfile = posixpath.join(
            self.yaml_dict['kaw_workdir'],
            self.yaml_dict['inv_name'],
            str(self.iteration),
            str(exp_num),
            logfile
        )
        sub_cmd = self.pipeline[exp_num][pipeline_step]
        cmd = "nohup %s > %s 2>&1 &\n echo $!" % (sub_cmd, abs_logfile)
        log.info("running command: %s" % cmd)
        self.client.exec_command("screen -r %s" % str(exp_num))  # reattach to an existing screen (one screen for each experiment)
        stdin, stdout, stderr = self.client.exec_command(cmd)
        pid = stdout.readlines()[0].strip()  # get the PID of the initiated process
        self.client.exec_command("screen -d")  # Detach the screen
        return pid

    def ping_process(self, pid):
        """
        Check if the process with Process ID=pid is still running.
        """
        stdin, stdout, stderr = self.client.exec_command("ps -o pid= -p %s" % pid)
        stdout_read = stdout.readlines()
        if len(stdout_read) > 0:
            return True
        else:
            return False

    def test_status(self):
        """
        Check the status of all running jobs on the server. If an experiment has run to completion,
        remove the experiment from the pid_dict.
        """
        if len(self.pid_dict) == 0:
            return False

        remove_from_dict = []
        for exp_num in self.pid_dict:
            pid = self.pid_dict[exp_num]["pid"]
            if self.ping_process(pid):  # Process is not done, wait...
                continue
            else:  # Process is done, continue with next one
                sub_process = self.pid_dict[exp_num]["subprocess"]
                if sub_process == len(self.pipeline['1']):  # All steps done
                    remove_from_dict.append(exp_num)  # save the exp to remove from the pid_dict
                    self.client.exec_command("screen -r %s" % exp_num)
                    self.client.exec_command("exit")  # Exit screen
                else:
                    next_subprocess = sub_process + 1
                    self.pid_dict[exp_num]["subprocess"] = next_subprocess
                    new_pid = self.run_command(exp_num, next_subprocess)
                    self.pid_dict[exp_num]["pid"] = new_pid

        if len(remove_from_dict) > 0:  # Remove any experiments that are done
            print "removing...", remove_from_dict
            for exp_num in remove_from_dict:
                del self.pid_dict[exp_num]

    def next_iteration(self):
        self.iteration += 1

    def run_pipeline(self):
        self.client = self.paramiko_connect()
        self.make_directories()
        self.initiate_experiments()
        self.paramiko_disconnect()
        time.sleep(self.check_interval)
        run_time = self.check_interval
        complete = False
        while not complete and run_time < self.stop_time:
            self.client = self.paramiko_connect()
            self.test_status()
            print "current uptime: %s seconds" % str(run_time)
            if len(self.pid_dict) == 0:  # All processes are done
                return

            self.paramiko_disconnect()
            time.sleep(self.check_interval)
            run_time += self.check_interval

    def collect_raw_result(self, filename):
        result_dict = {}
        self.client = self.paramiko_connect()
        for exp_num in range(1, self.num_exp+1):
            res_file = posixpath.join(
                self.yaml_dict['kaw_workdir'],
                self.yaml_dict['inv_name'],
                str(self.iteration),
                str(exp_num),
                filename
            )
            stdin, stdout, stderr = self.client.exec_command("cat %s" % res_file)
            out = stdout.readlines()
            result_dict[exp_num] = out

        self.paramiko_disconnect()
        return result_dict

    def calculate_result(self, raw_result, pd_worksheet):
        result_dict = {}
        result_types = ["TP", "FN", "FP", "F1"]
        exp_nums = range(1, len(self.pipeline)+1)
        for exp_num in exp_nums:
            result_dict[exp_num] = {}
            for result_type in result_types:
                result_dict[exp_num][result_type] = None

            for sub_result in raw_result[exp_num]:
                sub_result.strip()
                sub_result_split = sub_result.split("\t")
                sub_result_type = sub_result_split[0]
                sub_result_value = int(sub_result_split[1])
                result_dict[exp_num][sub_result_type] = sub_result_value

            # Calculate F1 score, the harmonic mean of precision and recall (sensitivity)
            TP = result_dict[exp_num]["TP"]
            FP = result_dict[exp_num]["FP"]
            FN = result_dict[exp_num]["FN"]
            precision = float(TP) / (TP + FP)
            recall = float(TP) / (TP + FN)
            F1 = 2 * (precision * recall) / (precision + recall)
            pd_worksheet["F1 Score"][exp_num] = F1
            pd_worksheet["True Positives"][exp_num] = TP
            pd_worksheet["False Positives"][exp_num] = FP
            pd_worksheet["False Negatives"][exp_num] = FN

        return pd_worksheet