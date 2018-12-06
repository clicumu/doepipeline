import logging
import os

from doepipeline.executor.local import LocalPipelineExecutor

# See https://slurm.schedmd.com/squeue.html for job state codes
OK_JOB_STATUS = (
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESV_DEL_HOLD",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUE",
    "RESIZING",
    "REVOKED",
    "SUSPENDED",
    "SPECIAL_EXIT",
    "SIGNALING"
)
FAIL_JOB_STATUS = (
    "FAILED",
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "STOPPED",
    "TIMEOUT"
)

# Specify the fields to request with sacct.
# Expand the 'State' field so we don't risk getting a truncated msg.
_field_len = 30
SACCT_FIELDS = [
    'JobID',
    'JobName',
    'Partition',
    'Account',
    'AllocCPUS',
    'State%{}'.format(_field_len),
    'ExitCode'
]


class SlurmPipelineExecutor(LocalPipelineExecutor):

    def run_jobs(self, job_steps, experiment_index, env_variables, **kwargs):
        try:
            slurm = kwargs['slurm']
        except KeyError:
            TypeError("Missing key-word argument: 'slurm'")

        slurm_command = 'sbatch {script}'

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

                command_step = slurm_command
            else:
                command_step = 'nohup {script} 2>&1 & echo $!'

            logging.info('Starts pipeline step: {}'.format(step_name))

            for exp_name, script in zip(experiment_index, step):
                current_workdir = os.path.join(self.workdir, str(exp_name))
                job_name = '{0}_exp_{1}'.format(step_name, exp_name)
                completed_flag_file = os.path.join(current_workdir, job_name + '.completed')

                if os.path.isfile(completed_flag_file) and self.recovery:
                    logging.info('The pipeline step {} is already completed '
                                 'for experiment {}, skipping.'.format(step_name, exp_name))
                else:
                    if slurm_spec is not None:
                        # Create SLURM-compatible batch-script file
                        # with current command.
                        batch_file = '{name}.sh'.format(name=job_name)
                        flag_lines = '\n'.join('#SBATCH {f}'.format(f=flag)
                                           for flag in flags)

                        file_script = "echo '#!/bin/sh\n{flags}\n{cmd}\n' > {batch_file}".format(
                            cmd=script, batch_file=batch_file, flags=flag_lines
                        )

                        self.touch_file(batch_file, cwd=current_workdir)
                        self.execute_command(file_script, job_name=exp_name, cwd=current_workdir)

                        command = command_step.format(script=batch_file)

                        # A little ugly work-around. Other executors watch the PID
                        # of the running process when executed with watch-keyword.
                        # To avoid this behaviour the job-name provided to SLURM is
                        # saved but the command is executed without setting watch
                        # to True.
                        completed_command = self.execute_command(
                            command,
                            job_name=exp_name,
                            cwd=current_workdir)

                        job_id = completed_command.stdout.strip().split()[-1].decode(self.encoding)
                        self.running_jobs[job_name] = {
                            'id': job_id,
                            'running_at_slurm': True,
                            'restarts': 2,
                            'command': command,
                            'exp_workdir': current_workdir,
                            'exp_name': exp_name
                        }

                    else:
                        # Jobs not running at SLURM are simply executed and
                        # pids are stored.
                        command = command_step.format(script=script)
                        completed_command = self.execute_command(
                            command,
                            job_name=exp_name,
                            cwd=current_workdir)
                        job_id = completed_command.stdout.strip().decode(self.encoding)
                        self.running_jobs[job_name] = {
                            'id': job_id,
                            'running_at_slurm': False,
                            'exp_workdir': current_workdir
                        }

            self.wait_until_current_jobs_are_finished()
            logging.info('Pipeline step finished: {}'.format(step_name))

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
            logging.debug('Polling "{}"'.format(job_name))
            is_running_slurm = job_info['running_at_slurm']
            if is_running_slurm:
                cmd = 'sacct -X -j {id} -o {fields}'.format(
                    id=job_info['id'], fields=','.join(SACCT_FIELDS))
                retries = 10
                check_returncode = True
            else:
                cmd = 'ps -a | grep {pid}'.format(pid=job_info['id'])
                # Grep returns exit code 1 if it can't find any matches.
                # Therefore, we should skip the error check in execute_command.
                retries = 1
                check_returncode = False

            completed_command = self.execute_command(
                cmd, attempts=retries, check=check_returncode)
            stdout = completed_command.stdout.decode(self.encoding)
            if is_running_slurm:
                status_rows = stdout.strip().split('\n')
                status_dict = dict(zip(status_rows[0].split(),
                                       status_rows[-1].split()))
                state = status_dict['State']

                if state == 'COMPLETED':
                    logging.info('{0} finished'.format(job_name))
                    # create the flag file for completed step "{job_name}.completed"
                    completed_filename = job_name + '.completed'
                    self.touch_file(completed_filename, cwd=job_info['exp_workdir'])
                    self.running_jobs.pop(job_name)

                elif state in FAIL_JOB_STATUS:
                    # State of job is either terminated or failed.
                    exit_code = status_dict['ExitCode']
                    msg = '{} has terminated or failed. (exit code {})'.format(
                        job_name,
                        exit_code)
                    logging.error(msg)
                    logging.error('Output from "{}":\n{}'.format(cmd, stdout))

                    # Ugly fix for random segmentation faults that makes it
                    # hard to get through the pipeline:
                    if exit_code == "11:0":
                        logging.error('Interpreting this as a segmentation fault. '
                                      'Attempting to restart the job.')
                        if job_info['restarts']:
                            self.running_jobs[job_name]['restarts'] -= 1
                            completed_command = self.execute_command(
                                job_info['command'],
                                job_name=job_info['exp_name'],
                                cwd=job_info['exp_workdir'])
                            job_id = completed_command.stdout.strip().split()[-1].decode(self.encoding)
                            self.running_jobs[job_name]['id'] = job_id
                            jobs_still_running.append(job_name)
                            logging.error('Successfully restarted the failed job.')
                            continue
                        else:
                            logging.error('Out of restart attempts.')

                    return self.JOB_FAILED, msg

                elif state in OK_JOB_STATUS:
                    # State of job is not failed and not completed,
                    # we should wait.
                    jobs_still_running.append(job_name)

                elif len(status_rows) == 2:
                    # A special case where the job is so fresh that sacct can't
                    # find the queried job. This results in only two rows
                    # returned and we should keep polling the job later.
                    jobs_still_running.append(job_name)

                else:
                    logging.error('Unknown job status "{}" from "{}"'.format(
                        state, cmd))
                    logging.error('Output from "{}":\n{}'.format(cmd, stdout))
                    return self.JOB_FAILED, "Unknown job status"

            else:  # Check status of process using ps.
                status = stdout.strip()
                if not status or 'done' in status.lower():
                    logging.info('{0} finished'.format(job_name))
                    # create the flag file for completed step "{job_name}.completed"
                    completed_filename = job_name + '.completed'
                    cmd = 'touch {}'.format(completed_filename)
                    completed_command = self.execute_command(
                        cmd, cwd=job_info['exp_workdir']
                    )
                    self.running_jobs.pop(job_name)
                elif 'exit' in status.lower():
                    msg = '{0} has failed'.format(job_name)
                    logging.error(msg)
                    return self.JOB_FAILED, msg
                else:
                    jobs_still_running.append(job_name)

        if jobs_still_running:
            msg = '{0} still running'.format(', '.join(jobs_still_running))
            return self.JOB_RUNNING, msg
        else:
            return self.JOB_FINISHED, 'no jobs running.'
