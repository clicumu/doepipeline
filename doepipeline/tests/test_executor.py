import unittest
import mock
import pandas as pd
import types
import time
import subprocess

from doepipeline.executor import BasePipelineExecutor, CommandError,\
    PipelineRunFailed, LocalPipelineExecutor
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

    def execute_command(self, command, watch=False, **kwargs):
        super(MockExecutor, self).execute_command(command, watch, **kwargs)
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


class TestBaseExecutorExecutions(ExecutorTestCase):

    def test_bad_command_input_raises_ValueError(self):
        executor = MockExecutor()
        bad_commands = [
            '',
            0,
            True,
            ['Commands'],
            {'command': 'again'},
            '   ',
            '\n',
            '\t',
        ]
        for bad_command in bad_commands:
            self.assertRaises(ValueError, executor.execute_command, bad_command)

    def test_watch_without_job_name_raises_ValueError(self):
        executor = MockExecutor()
        self.assertRaises(ValueError, executor.execute_command,
                          'command', watch=True)


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

    def test_failed_screen_run_poll_raises_PipelineRunFailed(self):
        executor = MockExecutor(run_in_batch=False, base_command='{script}')
        def poll_fail():
            return MockExecutor.JOB_FAILED, 'Error'

        executor.poll_jobs = poll_fail

        with self.assertRaises(PipelineRunFailed) as cm:
            executor.run_pipeline_collection(self.pipeline)
            self.assertEqual(cm.message, 'Error')

    @mock.patch('time.sleep')
    def test_running_screens_polls_again_and_breaks_when_finished(self,
                                                                  mock_sleep):
        executor = MockExecutor(run_in_batch=False,
                                base_command='{script}',
                                poll_interval=1)

        poll_checks = {'polls': 0, 'checked': False}
        def mock_poll():
            poll_checks['polls'] += 1
            if poll_checks['checked']:
                return MockExecutor.JOB_FINISHED, ''
            else:
                poll_checks['checked'] = True
                return MockExecutor.JOB_RUNNING, ''

        executor.poll_jobs = mock_poll
        executor.run_pipeline_collection(self.pipeline)
        self.assertGreaterEqual(poll_checks['polls'], 2)
        mock_sleep.assert_called_with(executor.poll_interval)


class TestBaseExecutorRunsBatches(ExecutorTestCase):

    def make_expected_scripts(self, workdir, base_script, executor, use_log=False):
        job1, job2 = self.design['Exp Id'].tolist()

        # Run-setup.
        expected_scripts = [
            'cd {}'.format(workdir),
            'mkdir {}'.format(job1),
            'mkdir {}'.format(job2)
        ]
        for step in range(2):
            prepared_jobs = list()
            for job, i in zip((job1, job2), (1, 2)):
                current_commands = ['cd {}'.format(job)]
                script_step = self.pipeline[job][step]

                if use_log:
                    log_file = executor.base_log.format(name=job, i=step + 1)
                    current_commands.append('touch {}'.format(log_file))
                    script = base_script.format(script=script_step,
                                                logfile=log_file)
                else:
                    script = base_script.format(script=script_step)

                current_commands += [script, 'cd ..']
                prepared_jobs.append(' && '.join(current_commands) + ' &')
            expected_scripts.append(' '.join(prepared_jobs))
            expected_scripts[-1] += ' wait'

        return expected_scripts

    def test_run_batch_without_log_gives_correct_output(self):
        for base_script in ('{script} &', '{script}'):
            executor = MockExecutor(run_in_batch=True, base_command=base_script)
            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = self.make_expected_scripts('.', '{script}', executor)
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_run_batch_with_log_gives_correct_output(self):
        base_command = '{script} > {logfile}'
        for command in (base_command, base_command + ' &'):
            executor = MockExecutor(run_in_batch=True,
                                    base_command=command)
            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = self.make_expected_scripts('.', base_command,
                                                          executor, use_log=True)
            self.maxDiff = None
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_failed_batch_run_poll_raises_PipelineRunFailed(self):
        executor = MockExecutor(run_in_batch=True, base_command='{script}')
        def poll_fail():
            return MockExecutor.JOB_FAILED, 'Error'

        executor.poll_jobs = poll_fail

        with self.assertRaises(PipelineRunFailed) as cm:
            executor.run_pipeline_collection(self.pipeline)
            self.assertEqual(cm.message, 'Error')

    @mock.patch('time.sleep')
    def test_running_screens_polls_again_and_breaks_when_finished(self,
                                                                  mock_sleep):
        executor = MockExecutor(run_in_batch=True,
                                base_command='{script}',
                                poll_interval=1)

        poll_checks = {'polls': 0, 'checked': False}
        def mock_poll():
            poll_checks['polls'] += 1
            if poll_checks['checked']:
                return MockExecutor.JOB_FINISHED, ''
            else:
                poll_checks['checked'] = True
                return MockExecutor.JOB_RUNNING, ''

        executor.poll_jobs = mock_poll
        executor.run_pipeline_collection(self.pipeline)
        self.assertGreaterEqual(poll_checks['polls'], 2)
        mock_sleep.assert_called_with(executor.poll_interval)


class TestLocalExecutor(ExecutorTestCase):

    @mock.patch('subprocess.Popen')
    def test_Popen_is_called_when_command_executes(self, mock_popen):
        executor = LocalPipelineExecutor()
        command = 'hello'
        executor.execute_command(command)
        mock_popen.assert_called_with(command, shell=True)

    @mock.patch('subprocess.Popen')
    def test_process_saved_when_command_called_with_watch(self, mock_popen):
        executor = LocalPipelineExecutor()
        command = 'hello'
        job_name = 'name'
        executor.execute_command(command, watch=True, job_name=job_name)
        mock_popen.assert_called_with(command, shell=True)
        self.assertIn(job_name, executor.running_jobs)
        self.assertIs(executor.running_jobs[job_name], mock_popen.return_value)

    @mock.patch('subprocess.Popen')
    def test_Popen_is_not_called_with_bad_command(self, mock_popen):
        executor = LocalPipelineExecutor()
        bad_commands = [
            '',
            0,
            True,
            ['Commands'],
            {'command': 'again'},
            '   ',
            '\n',
            '\t',
        ]
        for bad_command in bad_commands:
            self.assertRaises(ValueError, executor.execute_command, bad_command)
            self.assertFalse(mock_popen.called)

    @mock.patch('subprocess.Popen')
    def test_PipelineRunFailed_raised_when_subprocess_failed(self, mock_popen):
        mock_popen.return_value.poll.return_value = -1
        mock_popen.return_value.return_code = -1

        batch_executor = LocalPipelineExecutor(run_in_batch=True)
        self.assertRaises(PipelineRunFailed,
                          batch_executor.run_pipeline_collection,
                          self.pipeline)

        screen_executor = LocalPipelineExecutor()
        self.assertRaises(PipelineRunFailed,
                          screen_executor.run_pipeline_collection,
                          self.pipeline)

    @mock.patch('time.sleep')
    @mock.patch('subprocess.Popen')
    def test_job_are_finished_when_return_code_is_zero(self, mock_popen,
                                                       mock_sleep):
        mock_popen.return_value.poll.return_value = 0
        mock_popen.return_value.return_code = 0

        batch_executor = LocalPipelineExecutor(run_in_batch=True)
        batch_executor.run_pipeline_collection(self.pipeline)

        self.assertEqual(len(batch_executor.running_jobs), 0)
        self.assertFalse(mock_sleep.called)

        screen_executor = LocalPipelineExecutor()
        screen_executor.run_pipeline_collection(self.pipeline)

        self.assertEqual(len(screen_executor.running_jobs), 0)
        self.assertFalse(mock_sleep.called)

    @mock.patch('time.sleep')
    @mock.patch('subprocess.Popen')
    def test_jobs_are_polled_agained_when_poll_is_None(self, mock_popen,
                                                       mock_sleep):
        polled = {'yes': False, 'calls': 0}

        def poll():
            polled['calls'] += 1
            if polled['yes']:
                mock_popen.return_value.return_code = 0
                return 0
            else:
                polled['yes'] = True
                return None

        mock_popen.return_value.poll = poll

        batch_executor = LocalPipelineExecutor(run_in_batch=True)
        batch_executor.run_pipeline_collection(self.pipeline)

        # Called twice for first step and once for second.
        self.assertEqual(polled['calls'], 3)

        polled['yes'] = False
        polled['calls'] = 0

        screen_executor = LocalPipelineExecutor(run_in_batch=True)
        screen_executor.run_pipeline_collection(self.pipeline)

        # Called twice for first step and once for second.
        self.assertEqual(polled['calls'], 3)
