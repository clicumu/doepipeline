import logging
import re
from contextlib import contextmanager
from doepipeline.executor.base import BasePipelineExecutor,\
    PipelineRunFailed, CommandError

log = logging.getLogger(__name__)


class SerialExecutorMixin(BasePipelineExecutor):

    def __init__(self, *args, base_command=None, **kwargs):
        if base_command is None:
            base_command = '{script} > {logfile}'
        super(SerialExecutorMixin, self).__init__(*args,
                                                  base_command=base_command,
                                                  **kwargs)

    def run_jobs(self, job_steps, experiment_index, env_variables, **kwargs):
        """ Run all scripts using serial execution.

        I.e. no parallelism.

        :param job_steps: List of step-wise scripts.
        :type job_steps: list[list]
        :param experiment_index: List of job-names.
        :type experiment_index: list[str]
        :param env_variables: dictionary of environment variables to set.
        :type env_variables: dict
        """
        self._set_env_variables(env_variables)

        for i, step in enumerate(job_steps.values(), start=1):
            for script, job_name in zip(step, experiment_index):
                log_file = self.base_log.format(name=job_name, i=i)
                self._cd(job_name, job_name=job_name)
                try:
                    command = self.base_command.format(script=script)
                except KeyError:
                    has_log = True
                    command = self.base_command.format(script=script,
                                                       logfile=log_file)
                else:
                    has_log = False

                if has_log:
                    self._touch(log_file, job_name=job_name)

                try:
                    self.execute_command(command, wait=True, watch=True,
                                         job_name=job_name)
                except CommandError as e:
                    raise PipelineRunFailed(str(e))

                self._cd('..', job_name=job_name)

    def poll_jobs(self):
        """ Polling does nothing since commands are waited for """


class ScreenExecutorMixin(BasePipelineExecutor):

    def run_jobs(self, job_steps, experiment_index, env_variables, **kwargs):
        """ Run the collection of jobs in parallel using screens.

        Example job_steps:
        >>> job_steps = [
        ...     ['./script_1 --opt1 factor1_1', './script_1 --opt1 factor1_2'],
        ...     ['./script_2 --opt2 factor2_1', './script_2 --opt2 factor2_2']
        ... ]

        :param job_steps: List of step-wise scripts.
        :type job_steps: list[list]
        :param experiment_index: List of job-names.
        :type experiment_index: list[str]
        :param env_variables: dictionary of environment variables to set.
        :type env_variables: dict
        """
        log.info('Run jobs in parallel using screens')
        base_command = self.base_command

        # Look for correct ending.
        match = re.search(r'&\s*((?!$)\s*echo\s*\$!$|$)', base_command)
        if match is None:
            base_command += ' & echo $!'
        elif match.group().strip() == '&':
            base_command += ' echo $!'

        for job_name in experiment_index:
            log.debug('Setting up screen: {}'.format(job_name))
            self._make_screen(job_name)
            with self.screen(job_name):
                self._cd(job_name)
                if env_variables is not None:
                    self._set_env_variables(env_variables)

        # Run all scripts of each step in parallel using screens.
        # If a step fails, raise PipelineRunFailed.
        for i, step in enumerate(job_steps.values(), start=1):
            for script, job_name in zip(step, experiment_index):
                # Prepare log and command.
                log_file = self.base_log.format(name=job_name, i=i)

                try:
                    command = base_command.format(script=script)
                except KeyError:
                    has_log = True
                    command = base_command.format(script=script,
                                                  logfile=log_file)
                else:
                    has_log = False

                # Execute script in screen.
                with self.screen(job_name):
                    if has_log:
                        self._touch(log_file)
                    log.debug('Executes: {}'.format(command))
                    self.execute_command(command, watch=True, job_name=job_name)

            try:
                self.wait_until_current_jobs_are_finished()
            except PipelineRunFailed:
                raise

    def poll_jobs(self):
        """ Check job statuses in each screen.

        Jobs are checked by executing :code:`ps -a | grep [pid]` in
        each screen for each process-ID. If the resulting string contains
        "Done" the job is considered finished. If there is no resulting
        string or the resulting string contains "Exit" the job is considered
        failed. Else it is considered running.

        :return: status, message
        :rtype: tuple
        """
        still_running = list()
        for job_name, pid in self.running_jobs.items():
            cmd = 'ps -a | grep {pid}'.format(pid=pid)

            with self.screen(job_name):
                __, stdout, __ = self.execute_command(cmd)

            status = stdout.read().strip()
            if 'done' in status.lower():
                log.info('{0} finished'.format(job_name))
                self.running_jobs.pop(job_name)
            elif not status or 'exit' in status.lower():
                return self.JOB_FAILED, '{0} has failed'.format(job_name)
            else:
                still_running.append(job_name)

        if still_running:
            msg = '{0} still running'.format(', '.join(still_running))
            return self.JOB_RUNNING, msg
        else:
            return self.JOB_FINISHED, 'no jobs running.'

    @contextmanager
    def screen(self, screen_name):
        """ Context-manager to run commands within Linux-screens.

        :param screen_name: Name of screen to connect to.
        """
        self._reconnect_screen(screen_name)
        try:
            yield
        finally:
            self._disconnect_current_screen()

    def _disconnect_current_screen(self):
        self.execute_command('screen -d')

    def _reconnect_screen(self, name):
        self.execute_command('screen -r {}'.format(name))

    def _make_screen(self, name):
        self.execute_command('screen -S {}'.format(name))
        self._disconnect_current_screen()


class BatchExecutorMixin(BasePipelineExecutor):

    def poll_jobs(self):
        """ Check job statuses.

        Jobs are checked by executing :code:`ps -a | grep [pid]` for
        each process-ID. If the resulting string contains "Done" the
        job is considered finished. If there is no resulting string
        or the resulting string contains "Exit" the job is considered
        failed. Else it is considered running.

        :return: status, message
        :rtype: tuple
        """
        raise NotImplementedError
        # still_running = list()
        # for job_name, pid in self.running_jobs.items():
        #     cmd = 'ps -a | grep {pid}'.format(pid=pid)
        #     __, stdout, __ = self.execute_command(cmd)
        #     status = stdout.read().strip()
        #     if 'done' in status.lower():
        #         log.info('{0} finished'.format(job_name))
        #         self.running_jobs.pop(job_name)
        #     elif not status or 'exit' in status.lower():
        #         return self.JOB_FAILED, '{0} has failed'.format(job_name)
        #     else:
        #         still_running.append(job_name)
        #
        # if still_running:
        #     msg = '{0} still running'.format(', '.join(still_running))
        #     return self.JOB_RUNNING, msg
        # else:
        #     return self.JOB_FINISHED, 'no jobs running.'

    def run_jobs(self, job_steps, experiment_index, env_variables, **kwargs):
        """ Run the collection of jobs in parallel using batch execution.

        Example job_steps:
        >>> job_steps = [
        ...     ['./script_1 --opt1 factor1_1', './script_1 --opt1 factor1_2'],
        ...     ['./script_2 --opt2 factor2_1', './script_2 --opt2 factor2_2']
        ... ]

        Given above example, the following batch-scripts will be run::

            cd [exp_name1] && nohup ./script_1 --opt1 factor1_1 > [exp_name1]_step_1.log 2>&1 && cd .. &
            cd [exp_name2] && nohup ./script_1 --opt1 factor1_2 > [exp_name2]_step_1.log 2>&1 && cd .. &
            wait

        And when finished::

            cd [exp_name1] && nohup ./script_2 --opt2 factor2_1 > [exp_name1]_step_2.log 2>&1 && cd .. &
            cd [exp_name2] && nohup ./script_2 --opt2 factor2_2 > [exp_name2]_step_2.log 2>&1 && cd .. &
            wait

        :param job_steps: List of step-wise scripts.
        :type job_steps: list[list]
        :param experiment_index: List of job-names.
        :type experiment_index: list[str]
        :param env_variables: Environment variables to set.
        :type env_variables: dict
        """
        log.info('Run batch jobs')

        if self.base_command.endswith(' &'):
            base = self.base_command[:-2]
        else:
            base = self.base_command

        if env_variables is not None:
            self._set_env_variables(env_variables)

        base_command= 'cd {job_dir} && {{log_script}}' + base + ' && cd .. & echo $! &'

        for i, step in enumerate(job_steps.values(), start=1):
            commands = list()
            for script, job_name in zip(step, experiment_index):
                # Prepare log and command.
                log_file = self.base_log.format(name=job_name, i=i)

                try:
                    command = base_command.format(job_dir=job_name,
                                                  script=script)
                except KeyError:
                    command = base_command.format(job_dir=job_name,
                                                  script=script,
                                                  logfile=log_file)
                    log_script = 'touch {logfile} && '.format(logfile=log_file)
                    command = command.format(log_script=log_script)
                else:
                    command = command.format(log_script='')

                commands.append(command)

            commands.append('wait')
            batch_command = ' '.join(commands)
            self.execute_command(batch_command, watch=True,
                                 job_name=experiment_index)

            try:
                self.wait_until_current_jobs_are_finished()
            except PipelineRunFailed:
                raise


class SlurmExecutorMixin(BasePipelineExecutor):

    def run_jobs(self, job_steps, experiment_index, env_variables, slurm):
        slurm_command = 'sbatch -A {A} {flags} -J {{name}} {{script}}'

        for (step_name, step), slurm_spec in zip(job_steps.items(), slurm['jobs']):

            if slurm_spec is not None:
                flag_specs = ((key, value) for key, value in slurm_spec.items())
                flags = []
                for flag, value in flag_specs:
                    # Prepare flag-key first since some flags doesn't carry
                    # a parameter...
                    new_flag = ('-{f}' if len(flag) == 1 else '--{f}').format(f=flag)

                    # ... but if they do, add parameter.
                    if value is not None:
                        new_flag += ' {}'.format(value)
                    flags.append(new_flag)

                # Preformat command-string with non-step specific info.
                flag_str = ' '.join(flags)
                command_step = slurm_command.format(A=slurm['account_name'],
                                                   flags=flag_str)
            else:
                command_step = 'nohup {script} > {name}.log 2>&1 & echo $!'

            for exp_name, script in zip(experiment_index, step):
                job_name = '{0}_exp_{1}'.format(step_name, exp_name)

                if slurm_spec is not None:
                    # Create SLURM-compatible batch-script file
                    # with current command.
                    batch_file = '{name}.sh'.format(name=job_name)
                    file_script = 'echo "#!/bin/sh\n{cmd}\n" > {name}.sh'.format(
                        cmd=script, name=job_name
                    )
                    self.execute_command(file_script, job_name=exp_name)

                    command = command_step.format(name=job_name,
                                                  script=batch_file)

                    # A little ugly work-around. Other executors watch the PID
                    # of the running process when executed with watch-keyword.
                    # To avoid this behaviour the job-name provided to SLURM is
                    # saved but the command is executed without setting watch
                    # to True.
                    _, stdout, _ = self.execute_command(command, job_name=exp_name)
                    job_id = stdout.strip().split()[-1]
                    self.running_jobs[job_name] = {
                        'id': job_id, 'running_at_slurm': True
                    }

                else:
                    # Jobs not running at SLURM are simply executed and
                    # pids are stored.
                    command = command_step.format(script=script, name=job_name)
                    _, pid, _ = self.execute_command(command, job_name=exp_name)
                    self.running_jobs[job_name] = {
                        'id': pid.strip(), 'running_at_slurm': False
                    }

            self.wait_until_current_jobs_are_finished()

    def poll_jobs(self):
        """ Check job statuses.

        If job is started using SLURMS `sbatch` the job statuses are read
        by executing :code:`sacct -j <job_id>`. Otherwise
        the status is assessed by executing :code:`ps -a | grep <pid>`.

        :return: status, message
        :rtype: str, str
        """
        jobs_still_running = list()

        # Copy jobs to allow mutation of self.running_jobs.
        current_jobs = [job for job in self.running_jobs.items()]

        for job_name, job_info in current_jobs:
            is_running_slurm = job_info['running_at_slurm']
            if is_running_slurm:
                cmd = 'sacct -j {id}'.format(id=job_info['id'])
            else:
                cmd = 'ps -a | grep {pid}'.format(pid=job_info['id'])

            __, stdout, __ = self.execute_command(cmd)

            if is_running_slurm:
                status_rows = stdout.strip().split('\n')
                status_dict = dict(zip(status_rows[0].split()[-2:],
                                       status_rows[-1].split()[-2:]))

                if status_dict['State'] == 'FAILED':
                    exit_code = status_dict['ExitCode']
                    msg = '{0} has failed. (exit code {1})'.format(job_name,
                                                                   exit_code)
                    log.error(msg)
                    return self.JOB_FAILED, msg

                if status_dict['State'] == 'COMPLETED':
                    log.info('{0} finished'.format(job_name))
                    self.running_jobs.pop(job_name)

                else:
                    jobs_still_running.append(job_name)

            else:  # Check status of process using ps.
                status = stdout.strip()
                if not status or 'done' in status.lower():
                    log.info('{0} finished'.format(job_name))
                    self.running_jobs.pop(job_name)
                elif 'exit' in status.lower():
                    msg = '{0} has failed'.format(job_name)
                    log.error(msg)
                    return self.JOB_FAILED, msg
                else:
                    jobs_still_running.append(job_name)

        if jobs_still_running:
            msg = '{0} still running'.format(', '.join(jobs_still_running))
            return self.JOB_RUNNING, msg
        else:
            return self.JOB_FINISHED, 'no jobs running.'