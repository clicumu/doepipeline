"""
This module contains executors for pipeline execution remotely.
"""
import paramiko

from doepipeline.executor.base import BasePipelineExecutor


class BaseSSHExecutor(BasePipelineExecutor):

    def connect(self):
        pass

    def disconnect(self):
        pass

    def execute_command(self, command, watch=False, **kwargs):
        return super(BaseSSHExecutor, self).execute_command(command, watch,
                                                            **kwargs)

    def run_jobs(self, job_steps, experiment_index, env_variables):
        super(BaseSSHExecutor, self).run_jobs(job_steps, experiment_index,
                                              env_variables)

    def poll_jobs(self):
        super(BaseSSHExecutor, self).poll_jobs()