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
        return self.JOB_FINISHED, ''


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


class TestBaseExecutorSetup(ExecutorTestCase):

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
            self.assertRaises(AssertionError, MockExecutor, **dict([input]))

    def test_invalid_base_script_raises_ValueError(self):
        self.assertRaises(ValueError, MockExecutor,
                          base_command='{script}_{bad}')
        self.assertRaises(ValueError, MockExecutor,
                          base_command='no_script_{logfile}')
        self.assertRaises(ValueError, MockExecutor, base_command='no_script')

    def test_invalid_base_log_raises_ValueError(self):
        self.assertRaises(ValueError, MockExecutor, base_log='{bad_tag}_{i}')
        self.assertRaises(ValueError, MockExecutor, base_log='{name}_{bad_tag}')
        self.assertRaises(ValueError, MockExecutor, base_log='{name}_no_i')
        self.assertRaises(ValueError, MockExecutor, base_log='{i}_no_name')

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



class TestBaseExecutorRunsScreens(ExecutorTestCase):

    def make_expected_scripts(self, workdir, base_script, executor, use_log=False):
        job1, job2 = self.design['Exp Id'].tolist()

        # Run-setup.
        expected_scripts = [
            'cd {}'.format(workdir),
            'mkdir {}'.format(job1),
            'mkdir {}'.format(job2)
        ]
        # Set-up screens
        for job in (job1, job2):
            expected_scripts += [
                'screen -S {}'.format(job),
                'screen -d',
                'screen -r {}'.format(job),
                'cd {}'.format(job),
                'screen -d'
            ]

        # Screen script runs.
        for step in range(2):
            for job, i in zip((job1, job2), (1, 2)):
                expected_scripts += ['screen -r {}'.format(job)]

                if use_log:
                    log_file = executor.base_log.format(name=job, i=step + 1)
                    expected_scripts += ['touch {}'.format(log_file)]
                    script = base_script.format(script=self.pipeline[job][step],
                                                logfile=log_file)
                else:
                    script = base_script.format(script=self.pipeline[job][step])

                script += ' &'
                expected_scripts += [
                    script,
                    'screen -d'
                ]
        return expected_scripts

    def test_run_jobs_without_logs_give_correct_output(self):
        base_cmd = '{script}'
        executor = MockExecutor(run_in_batch=False, base_command=base_cmd)
        executor.run_pipeline_collection(self.pipeline)
        expected_scripts = self.make_expected_scripts('.', base_cmd,
                                                      executor, False)
        self.assertListEqual(expected_scripts, executor.scripts)

    def test_run_jobs_with_logs_give_corrext_output(self):
        base_cmd = '{script}_{logfile}'
        executor = MockExecutor(run_in_batch=False,
                                base_command=base_cmd)
        executor.run_pipeline_collection(self.pipeline)
        expected_scripts = self.make_expected_scripts('.', base_cmd,
                                                      executor, True)
        self.assertListEqual(expected_scripts, executor.scripts)


class TestBaseExecutorRunsBatches(ExecutorTestCase):

    def test_job_run_in_batch(self):
        executor = MockExecutor(base_command='{script}')
        executor.run_pipeline_collection(self.pipeline)
        job1, job2 = self.design['Exp Id'].tolist()

        # Run-setup.
        expected_scripts = [
            'cd {}'.format(executor.workdir),
            'mkdir {}'.format(job1),
            'mkdir {}'.format(job2)
        ]
        for step in range(2):
            for job, i in zip((job1, job2), (1, 2)):
                pass