"""
This module contains executors for simple pipeline execution in
a Linux-shell.
"""
import subprocess
import os
from collections import OrderedDict

from .base import BasePipelineExecutor, CommandError, PipelineRunFailed


class LocalPipelineExecutor(BasePipelineExecutor):
    """
    Executor class running pipeline locally in a linux shell.
    """
    def __init__(self, *args, base_command=None, run_serial=True, **kwargs):
        if base_command is None:
            base_command = '{script} > {logfile}'
        super(LocalPipelineExecutor, self).__init__(*args,
                                                    base_command=base_command,
                                                    **kwargs)
        self.run_serial = run_serial
        self.running_jobs = dict()

    def poll_jobs(self):
        still_running = list()
        for job_name, process in self.running_jobs.items():
            if process.poll() is None:
                still_running.append(job_name)
            else:
                if process.return_code != 0:
                    return self.JOB_FAILED, '{} has failed'.format(job_name)
                else:
                    self.running_jobs.pop(job_name)

        if still_running:
            msg = '{} still running'.format(', '.join(still_running))
            return self.JOB_RUNNING, msg
        else:
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
        if watch:
            try:
                process = subprocess.Popen(command, shell=True)
            except OSError as e:
                raise CommandError(str(e))
            self.running_jobs[kwargs.pop('job_name')] = process

            if wait:
                process.wait()
        else:
            try:
                # Note: This will wait until execution finished.
                subprocess.call(command)
            except OSError as e:
                raise CommandError(str(e))

    def read_file_contents(self, file_name, **kwargs):
        """ Read contents of local file.

        :param str file_name: File to read.
        :return: File contents.
        :rtype: str
        """
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
            for script, job_name in zip(step, experiment_index):
                log_file = self.base_log.format(name=job_name, i=i)
                self.change_dir(job_name, job_name=job_name)
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
                                         watch=True, job_name=job_name)
                except CommandError as e:
                    raise PipelineRunFailed(str(e))

                self.change_dir('..', job_name=job_name)

    def touch_file(self, file_name, times=None):
        with open(file_name, 'a'):
            os.utime(file_name, times=times)

    def make_dir(self, dir, **kwargs):
        os.makedirs(dir, **kwargs)

    def change_dir(self, dir, **kwargs):
        os.chdir(dir)

    def set_env_variables(self, env_variables):
        for key, value in env_variables.items():
            os.environ[key] = value