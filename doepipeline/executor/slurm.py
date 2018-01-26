import logging

from doepipeline.executor.local import LocalPipelineExecutor

log = logging.getLogger(__name__)


class SlurmPipelineExecutor(LocalPipelineExecutor):

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