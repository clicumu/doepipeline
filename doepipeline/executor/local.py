"""
This module contains executors for simple pipeline execution in
a Linux-shell.
"""
import subprocess
import os

from .base import BasePipelineExecutor, CommandError
from .mixins import BatchExecutorMixin, ScreenExecutorMixin, SerialExecutorMixin


class BaseLocalExecutor(BasePipelineExecutor):

    """
    Executor class running pipeline locally in a linux shell.
    """
    def __init__(self, run_in_batch=False, *args, **kwargs):
        super(BaseLocalExecutor, self).__init__(*args, **kwargs)
        assert isinstance(run_in_batch, bool), 'run_in_batch must be boolean'

        self.run_in_batch = run_in_batch
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
        super(BaseLocalExecutor, self).execute_command(command, watch,
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


class LocalBatchExecutor(BatchExecutorMixin, BaseLocalExecutor):

    def poll_jobs(self):
        return BaseLocalExecutor.poll_jobs(self)


class LocalScreenExecutor(ScreenExecutorMixin, BaseLocalExecutor):

    def poll_jobs(self):
        return BaseLocalExecutor.poll_jobs(self)


class LocalSerialExecutor(SerialExecutorMixin, BasePipelineExecutor):
    """ Executor class which runs jobs serially locally. """

    def execute_command(self, command, watch=False, wait=False, **kwargs):
        try:
            subprocess.call(command, shell=True)
        except Exception as e:
            raise CommandError('"{0}": {1}'.format(command, str(e)))

    def read_file_contents(self, file_name):
        """ Read contents of local file.

        :param str file_name: File to read.
        :return: File contents.
        :rtype: str
        """
        with open(file_name) as f:
            contents = f.read()

        return contents

    def _cd(self, dir):
        os.chdir(dir)


def LocalExecutor(*args, execution_type='serial', **kwargs):
    if execution_type == 'serial':
        return LocalSerialExecutor(*args, **kwargs)

    elif execution_type == 'screen':
        return LocalScreenExecutor(*args, **kwargs)

    elif execution_type == 'batch':
        return LocalBatchExecutor(*args, **kwargs)

    else:
        raise ValueError('unknown execution_type: {0}'.format(execution_type))