import os
import unittest
from unittest import mock
import pandas as pd
import copy
from collections import Sequence

from doepipeline.executor.base import BasePipelineExecutor
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
        return 'in', 'out', 'error'

    def poll_jobs(self):
        return self.JOB_FINISHED, ''

    def read_file_contents(self, file_name, directory=None):
        return 'A,1\nB,2'


class ExecutorTestCase(unittest.TestCase):

    executor_class = MockBaseExecutor
    init_args = tuple()
    init_kwargs = dict()

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
        design_spec = {
            'type': 'CCC',
            'factors': {
                'FactorA': {
                    'min': 0,
                    'max': 1,
                    'low_init': 0,
                    'high_init': .2
                },
                'FactorB': {
                    'min': 0,
                    'max': 1,
                    'low_init': .1,
                    'high_init': .3
                }
            },
            'responses': {
                'ResponseA': {'criterion': 'maximize'}
            }
        }

        self.work_dir = os.path.join(os.getcwd(), 'work_dir')
        os.makedirs(self.work_dir, exist_ok=True)

        self.env_vars = {'MYPATH': '~/a/path'}
        before = {'environment_variables': self.env_vars}
        self.outfile = 'my_results.txt'
        self.config = {
            'working_directory': self.work_dir,
            'design': design_spec,
            'results_file': self.outfile,
            'before_run': before,
            'pipeline': ['ScriptOne', 'ScriptTwo'],
            'ScriptOne': script_one,
            'ScriptTwo': script_two
        }
        self.design = pd.DataFrame([
            ['One', .1, .2],
            ['Two', .3, .4]
        ], columns=['Exp Id', 'FactorA', 'FactorB'])
        self.generator = PipelineGenerator(copy.deepcopy(self.config))
        self.pipeline = self.generator.new_pipeline_collection(self.design,'Exp Id')

    @mock.patch('os.makedirs')
    def test_execute_commands_returns_tuple(self, *args):
        executor = self.executor_class(*self.init_args, **self.init_kwargs)
        result = executor.execute_command('test')
        self.assertIsInstance(result, Sequence)
        self.assertEqual(len(result), 3)
        for value in result:
            self.assertIsInstance(value, str)