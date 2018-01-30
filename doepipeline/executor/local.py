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
        for job_name, process in dict(self.running_jobs).items():
            logging.debug('Polls "{}"'.format(job_name))
            if process.poll() is None:
                still_running.append(job_name)
            else:
                if process.returncode != 0:
                    logging.info('Job "{}" failed'.format(job_name))
                    return self.JOB_FAILED, '{} has failed'.format(job_name)
                else:
                    logging.info('Job "{}" finished'.format(job_name))
                    self.running_jobs.pop(job_name)

        if still_running:
            msg = '{} still running'.format(', '.join(still_running))
            return self.JOB_RUNNING, msg
        else:
            logging.info('All jobs finished.')
            return self.JOB_FINISHED, 'no jobs running.'

    def execute_command(self, command, watch=False, wait=False, **kwargs):
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
        if watch:
            try:
                process = subprocess.Popen(command, shell=True, **kwargs)
            except OSError as e:
                raise CommandError(str(e))

            self.running_jobs[job_name] = process
            if wait:
                process.wait()
        else:
            try:
                # Note: This will wait until execution finished.
                subprocess.call(command)
            except OSError as e:
                logging.warning('Command failed: "{}"'.format(command))
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

        for i, step in enumerate(job_steps.values(), start=1):
            logging.info('Starts pipeline step: {}'.format(step))
            for script, job_name in zip(step, experiment_index):
                current_workdir = os.path.join(self.workdir, job_name)

                log_file = self.base_log.format(name=job_name, i=i)

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
            logging.info('Pipeline step finished: {}'.format(step))

    def touch_file(self, file_name, times=None):
        logging.debug('Creates file: {}'.format(file_name))
        with open(file_name, 'a'):
            os.utime(file_name, times=times)

    def make_dir(self, dir, **kwargs):
        logging.debug('Make directory: {} (kwargs {})'.format(dir, kwargs))
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