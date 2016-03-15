import unittest
import pandas as pd
import copy
from collections import Sequence

from doepipeline.executor.base import BasePipelineExecutor
from doepipeline.executor.mixins import ScreenExecutorMixin, BatchExecutorMixin
from doepipeline.generator import PipelineGenerator


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
    pass


class MockScreenExecutor(ScreenExecutorMixin, MockBaseExecutor):
    pass


class ExecutorTestCase(unittest.TestCase):

    executor_class = MockBaseExecutor

    def setUp(self):
        script_one = {
            'script': './script_a',
            'factors': {
                'FactorA': {
                    'factor_name': 'Factor A',
                    'script_option': '--option'
                }
            }
        }
        script_two = {
            'script': './script_b {% FactorB %}',
            'factors': {
                'FactorB': {
                    'factor_name': 'Factor B',
                    'substitute': True
                }
            }
        }

        self.env_vars = {'MYPATH': '~/a/path'}
        before = {'environment_variables': self.env_vars}
        self.config = {
            'before_run': before,
            'pipeline': ['ScriptOne', 'ScriptTwo'],
            'ScriptOne': script_one,
            'ScriptTwo': script_two
        }
        self.design = pd.DataFrame([
            ['One', .1, .2],
            ['Two', .3, .4]
        ], columns=['Exp Id', 'Factor A', 'Factor B'])
        self.generator = PipelineGenerator(copy.deepcopy(self.config))
        self.pipeline = self.generator.new_pipeline_collection(self.design,'Exp Id')

    def test_execute_commands_returns_tuple(self):
        executor = self.executor_class()
        result = executor.execute_command('test')
        self.assertIsInstance(result, Sequence)
        self.assertEqual(len(result), 3)
        for value in result:
            self.assertIsInstance(value, str)