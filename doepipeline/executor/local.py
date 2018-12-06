"""
This module contains executors for simple pipeline execution in
a Linux-shell.
"""
import subprocess
import os
import logging
from collections import OrderedDict

from .base import BasePipelineExecutor, CommandError, PipelineRunFailed


class LocalPipelineExecutor(BasePipelineExecutor):
    """
    Executor class running pipeline locally in a linux shell.
    """
    def __init__(self, *args, base_command=None, run_serial=True, **kwargs):
        if base_command is None:
            base_command = '{script}'
        super(LocalPipelineExecutor, self).__init__(*args,
                                                    base_command=base_command,
                                                    **kwargs)
        self.run_serial = run_serial
        self.running_jobs = dict()

    def poll_jobs(self):
        still_running = list()
        for job_name, job_info in dict(self.running_jobs).items():
            logging.debug('Polls "{}"'.format(job_name))
            process = job_info['pid']
            if process.poll() is None:
                still_running.append(job_name)
            else:
                if process.returncode != 0:
                    logging.info('Job "{}" failed'.format(job_name))
                    return self.JOB_FAILED, '{} has failed'.format(job_name)
                else:
                    logging.info('Job "{}" finished'.format(job_name))
                    # create the flag file for completed step "{job_name}.completed"
                    completed_filename = job_name + '.completed'
                    self.touch_file(completed_filename,
                                    cwd=job_info['exp_workdir'])
                    self.running_jobs.pop(job_name)

        if still_running:
            msg = '{} still running'.format(', '.join(map(str, still_running)))
            return self.JOB_RUNNING, msg
        else:
            logging.info('All current jobs finished.')
            return self.JOB_FINISHED, 'no jobs running.'

    def execute_command(self, command, watch=False, wait=False,
                        check=True, attempts=3, **kwargs):
        """ Execute given command by executing it in subprocess.

        Calls are made using `subprocess`-module like::

            process = subprocess.Popen(command, shell=True)

        :param str command: Command to execute.
        :param bool watch: If True, monitor process.
        :param kwargs: Keyword-arguments.
        """
        super(LocalPipelineExecutor, self).execute_command(command, watch,
                                                           **kwargs)
        job_name = kwargs.pop('job_name', None)
        workdir = kwargs.get('cwd', '.')
        if watch:
            try:
                process = subprocess.Popen(command, shell=True, **kwargs)
            except OSError as e:
                raise CommandError(str(e))

            self.running_jobs[job_name] = {
                'pid': process,
                'exp_workdir': workdir
            }
            if wait:
                process.wait()
        else:
            while attempts:
                try:
                    # Note: This will wait until execution finished.
                    attempts -= 1
                    completed_process = subprocess.run(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=check,
                        **kwargs)
                    return completed_process
                except subprocess.CalledProcessError as e:
                    logging.error('Command failed:\n"{}"'.format(command))
                    logging.error('Output from subprocess.run():\n{}'.format(e))
                    if attempts:
                        logging.error('Retrying...')
                        continue
                    raise CommandError(str(e))

    def read_file_contents(self, file_name, directory=None, **kwargs):
        """ Read contents of local file.

        :param str file_name: File to read.
        :return: File contents.
        :rtype: str
        """
        if directory is not None:
            file_name = os.path.join(directory, file_name)

        logging.debug('Reads {}'.format(file_name))
        with open(file_name) as f:
            contents = f.read()

        return contents

    def run_jobs(self, job_steps, experiment_index, env_variables, **kwargs):
        """ Run all scripts.

        :param job_steps: List of step-wise scripts.
        :type job_steps: OrderedDict[key, list]
        :param experiment_index: List of job-names.
        :type experiment_index: list[str]
        :param env_variables: dictionary of environment variables to set.
        :type env_variables: dict
        """
        assert isinstance(job_steps, OrderedDict), 'job_steps must be ordered'
        self.set_env_variables(env_variables)

        for i, pipeline_step in enumerate(job_steps, start=1):
            logging.info('Starts pipeline step: {}'.format(pipeline_step))
            scripts = job_steps[pipeline_step]
            for script, exp_idx in zip(scripts, experiment_index):
                current_workdir = os.path.join(self.workdir, str(exp_idx))
                log_file = self.base_log.format(name=exp_idx, i=i)
                completed_flag_file_name = '_'.join([pipeline_step, str(exp_idx)]) + '.completed'
                completed_flag_file = os.path.join(current_workdir,
                                                   completed_flag_file_name)

                if os.path.isfile(completed_flag_file) and self.recovery:
                    logging.info('The pipeline step {} is already completed '
                                 'for experiment {}, skipping.'.format(pipeline_step, exp_idx))
                else:
                    job_name = '_'.join([pipeline_step, str(exp_idx)])
                    try:
                        command = self.base_command.format(script=script)
                    except KeyError:
                        has_log = True
                        command = self.base_command.format(script=script,
                                                           logfile=log_file)
                    else:
                        has_log = False

                    if has_log:
                        self.touch_file(log_file)
                    try:
                        self.execute_command(command, wait=self.run_serial,
                                             watch=True, job_name=job_name,
                                             cwd=current_workdir)
                    except CommandError as e:
                        raise PipelineRunFailed(str(e))

            self.wait_until_current_jobs_are_finished()
            logging.info('Pipeline step finished: {}'.format(pipeline_step))

    def make_dir(self, dir, **kwargs):
        logging.debug('Make directory: {} (kwargs {})'.format(dir, kwargs))
        if os.path.isdir(dir):
            logging.warning('Directory already exists. {}'.format(dir))
            return
        try:
            os.makedirs(dir, **kwargs)
        except (OSError, FileExistsError) as e:
            logging.warning('Failed directory creation: {}'.format(dir))
            raise CommandError(str(e))

    def change_dir(self, dir, **kwargs):
        logging.debug('Change directory: {} (kwargs {})'.format(dir, kwargs))
        try:
            os.chdir(dir)
        except (OSError, FileNotFoundError) as e:
            logging.warning('Failed directory change: {}'.format(dir))
            raise CommandError(str(e))

    def set_env_variables(self, env_variables):
        if env_variables:
            assert isinstance(env_variables, dict), 'env_variables must be dict'
            for key, value in env_variables.items():
                logging.debug('Sets env-variable: {}={}'.format(key, value))
                os.environ[key] = value
