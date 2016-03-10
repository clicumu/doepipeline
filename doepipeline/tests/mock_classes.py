from doepipeline.executor.base import BasePipelineExecutor
from doepipeline.executor.mixins import ScreenExecutorMixin, BatchExecutorMixin


class MockBaseExecutor(BasePipelineExecutor):

    """
    Mock-implementation of :class:`BasePipelineExecutor` which
    stores all called command in list.

    Used to test functionality of :class:`BasePipelineExecutor`
    """

    def __init__(self, *args, **kwargs):
        super(MockBaseExecutor, self).__init__(*args, **kwargs)
        self.scripts = list()

    def run_jobs(self, job_steps, experiment_index, env_variables):
        pass

    def execute_command(self, command, watch=False, **kwargs):
        super(MockBaseExecutor, self).execute_command(command, watch, **kwargs)
        self.scripts.append(command)

    def poll_jobs(self):
        return self.JOB_FINISHED, ''


class MockBatchExecutor(BatchExecutorMixin, MockBaseExecutor):

    def run_jobs(self, *args, **kwargs):
        self.run_batches(*args, **kwargs)


class MockScreenExecutor(ScreenExecutorMixin, MockBaseExecutor):

    def run_jobs(self, *args, **kwargs):
        self.run_in_screens(*args, **kwargs)