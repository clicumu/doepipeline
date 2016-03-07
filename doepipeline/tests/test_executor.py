import unittest
import pandas as pd
import types

from doepipeline.executor import BasePipelineExecutor, CommandError
from doepipeline.generator import PipelineGenerator


class MockExecutor(BasePipelineExecutor):

    """
    Mock-implementation of :class:`BasePipelineExecutor` which
    stores all called command in list.

    Used to test functionality of :class:`BasePipelineExecutor`
    """

    def __init__(self, *args, **kwargs):
        super(MockExecutor, self).__init__(*args, **kwargs)
        self.scripts = list()

    def connect(self, *args, **kwargs):
        pass

    def disconnect(self, *args, **kwargs):
        pass

    def execute_command(self, command):
        self.scripts.append(command)

    def poll_jobs(self):
        return self.JOB_FINISHED


class ExecutorTestCase(unittest.TestCase):

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
        self.config = {
            'pipeline': ['ScriptOne', 'ScriptTwo'],
            'ScriptOne': script_one,
            'ScriptTwo': script_two
        }
        self.design = pd.DataFrame([
            ['One', .1, .2],
            ['Two', .3, .4]
        ], columns=['Exp Id', 'Factor A', 'Factor B'])
        self.generator = PipelineGenerator(self.config)
        self.pipeline = self.generator.new_pipeline_collection(self.design,'Exp Id')


class TestBaseExecutor(ExecutorTestCase):

    def test_creation_doesnt_crash(self):
        executor = MockExecutor()

    def test_bad_creation_does_crash(self):
        bad_input = [
            ('workdir', 123),
            ('workdir', ''),
            ('workdir', True),
            ('run_in_batch', 'true'),
            ('run_in_batch', 1),
            ('run_in_batch', 0),
            ('run_in_batch', None),
            ('poll_interval', 0),
            ('poll_interval', -1),
            ('poll_interval', 'number'),
            ('poll_interval', True)
        ]
        for input in bad_input:
            try:
                self.assertRaises(AssertionError,
                                  lambda: MockExecutor(**dict([input])))
            except AssertionError, error:
                error.args += ('input {}'.format(input), )
                raise

    def test_run_pipeline_sets_up_properly(self):

        for workdir in (None, 'foldername', 'path/to/folder'):
            executor = MockExecutor(run_in_batch=False, workdir=workdir)

            # Monkey patch out screen-run to truncate script output.
            executor.run_in_screens = lambda steps, index: None

            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = [
                'cd {}'.format(workdir if workdir is not None else '.'),
                'mkdir {}'.format(self.design['Exp Id'][0]),
                'mkdir {}'.format(self.design['Exp Id'][1])
            ]
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_run_pipeline_sets_up_properly_when_no_workdir(self):
        for workdir in (None, 'foldername', 'path/to/folder'):
            executor = MockExecutor(run_in_batch=False, workdir=workdir)

            # Monkey patch out screen-run to truncate script output.
            executor.run_in_screens = lambda steps, index: None

            # Monkey patch to emulate failed cd
            has_run = {'yes': False}

            def _cd(self, path):
                self.execute_command('cd {}'.format(path))
                if not has_run['yes']:
                    has_run['yes'] = True
                    raise CommandError

            executor._cd = types.MethodType(_cd, executor)

            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = [
                'cd {}'.format(workdir if workdir is not None else '.'),
                'mkdir {}'.format(workdir if workdir is not None else '.'),
                'cd {}'.format(workdir if workdir is not None else '.'),
                'mkdir {}'.format(self.design['Exp Id'][0]),
                'mkdir {}'.format(self.design['Exp Id'][1])
            ]
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_job_steps_are_prepared_properly(self):
        executor = MockExecutor(run_in_batch=False)

        # Monkey patch out screen-run to simply return output.
        output = dict()
        def mock_run_in_screens(steps, index):
            output['steps'] = steps
            output['index'] = index

        executor.run_in_screens = mock_run_in_screens

        executor.run_pipeline_collection(self.pipeline)
        self.assertListEqual(output['index'], self.design['Exp Id'].values.tolist())

        i1, i2 = output['index']
        expected_steps = [
            [self.pipeline[i1][0], self.pipeline[i2][0]],
            [self.pipeline[i1][1], self.pipeline[i2][1]]
        ]
        self.assertListEqual(expected_steps, output['steps'])

    def test_job_run_in_screens(self):
        executor = MockExecutor(run_in_batch=False)

        executor.run_pipeline_collection(self.pipeline)

        # Run-setup.
        expected_scripts = [
            'cd {}'.format(executor.workdir),
            'mkdir {}'.format(self.design['Exp Id'][0]),
            'mkdir {}'.format(self.design['Exp Id'][1])
        ]


