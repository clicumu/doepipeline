import types
try:
    from unittest import mock
except ImportError:
    import mock

from doepipeline.executor.base import CommandError, PipelineRunFailed
from doepipeline.executor import LocalPipelineExecutor
from doepipeline.tests.executor_utils import  *


class TestBaseExecutorSetup(ExecutorTestCase):

    def test_creation_doesnt_crash(self):
        executor = MockBaseExecutor()

    def test_bad_creation_does_crash(self):
        bad_input = [
            ('workdir', 123),
            ('workdir', ''),
            ('workdir', True),
            ('poll_interval', 0),
            ('poll_interval', -1),
            ('poll_interval', 'number'),
            ('poll_interval', True)
        ]
        for input in bad_input:
            self.assertRaises(AssertionError, MockBaseExecutor, **dict([input]))

    def test_invalid_base_script_raises_ValueError(self):
        self.assertRaises(ValueError, MockBaseExecutor,
                          base_command='{script}_{bad}')
        self.assertRaises(ValueError, MockBaseExecutor,
                          base_command='no_script_{logfile}')
        self.assertRaises(ValueError, MockBaseExecutor, base_command='no_script')

    def test_invalid_base_log_raises_ValueError(self):
        self.assertRaises(ValueError, MockBaseExecutor, base_log='{bad_tag}_{i}')
        self.assertRaises(ValueError, MockBaseExecutor, base_log='{name}_{bad_tag}')
        self.assertRaises(ValueError, MockBaseExecutor, base_log='{name}_no_i')
        self.assertRaises(ValueError, MockBaseExecutor, base_log='{i}_no_name')

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    def test_run_pipeline_sets_up_properly(self, *args):
        executor = MockBaseExecutor(workdir=self.work_dir)

        executor.run_pipeline_collection(self.pipeline)
        expected_scripts = [
            'cd {}'.format(executor.workdir),
            'mkdir {}'.format(self.design['Exp Id'][0]),
            'mkdir {}'.format(self.design['Exp Id'][1])
        ]
        self.assertListEqual(expected_scripts, executor.scripts)

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    def test_run_pipeline_sets_up_properly_when_no_workdir(self, *args):
        for workdir in (None, 'foldername', 'path/to/folder'):
            executor = MockBaseExecutor(workdir=workdir)

            # Monkey patch to emulate failed cd
            has_run = {'yes': False}

            def _cd(self, path):
                self.execute_command('cd {}'.format(path))
                if not has_run['yes']:
                    has_run['yes'] = True
                    raise CommandError

            executor.change_dir = types.MethodType(_cd, executor)

            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = [
                'cd {}'.format(executor.workdir),
                'mkdir {}'.format(executor.workdir),
                'cd {}'.format(executor.workdir),
                'mkdir {}'.format(self.design['Exp Id'][0]),
                'mkdir {}'.format(self.design['Exp Id'][1])
            ]
            self.assertListEqual(expected_scripts, executor.scripts)

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    def test_job_steps_are_prepared_properly(self, *args):
        executor = MockBaseExecutor()

        # Monkey patch out screen-run to simply return output.
        output = dict()
        def mock_run_in_screens(steps, index, envs):
            output['steps'] = steps
            output['index'] = index

        executor.run_jobs = mock_run_in_screens

        executor.run_pipeline_collection(self.pipeline)
        self.assertListEqual(output['index'], self.design['Exp Id'].values.tolist())

        i1, i2 = output['index']
        expected_steps = [
            [self.pipeline[i1][0], self.pipeline[i2][0]],
            [self.pipeline[i1][1], self.pipeline[i2][1]]
        ]
        self.assertSequenceEqual(expected_steps, list(output['steps'].values()))


class TestBaseExecutorExecutions(ExecutorTestCase):

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    def test_bad_command_input_raises_ValueError(self, *args):
        executor = MockBaseExecutor()
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
        executor = MockBaseExecutor()
        self.assertRaises(ValueError, executor.execute_command,
                          'command', watch=True)


class TestLocalExecutor(ExecutorTestCase):

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    @mock.patch('subprocess.Popen')
    def test_Popen_is_called_when_command_executes(self, mock_popen, *args):
        executor = LocalPipelineExecutor()
        command = 'hello'
        executor.execute_command(command, watch=False)
        mock_popen.assert_called_with(command)

        executor.execute_command(command, job_name='job', watch=True)
        mock_popen.assert_called_with(command, shell=True)

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    @mock.patch('subprocess.Popen')
    def test_process_saved_when_command_called_with_watch(self, mock_popen, *args):
        executor = LocalPipelineExecutor()
        command = 'hello'
        job_name = 'name'
        executor.execute_command(command, watch=True, job_name=job_name)
        mock_popen.assert_called_with(command, shell=True)
        self.assertIn(job_name, executor.running_jobs)
        self.assertIs(executor.running_jobs[job_name], mock_popen.return_value)

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    @mock.patch('subprocess.Popen')
    def test_PipelineRunFailed_raised_when_subprocess_failed(self,
                                                             mock_popen,
                                                             *args):
        mock_popen.return_value.poll.return_value = -1
        mock_popen.return_value.return_code = -1

        executor = LocalPipelineExecutor()
        self.assertRaises(PipelineRunFailed,
                          executor.run_pipeline_collection,
                          self.pipeline)

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    @mock.patch('subprocess.Popen')
    def test_Popen_is_not_called_with_bad_command(self, mock_popen, *args):
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

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    @mock.patch('time.sleep')
    @mock.patch('subprocess.Popen')
    def test_job_are_finished_when_return_code_is_zero(self, mock_popen,
                                                       mock_sleep, *args):
        mock_popen.return_value.poll.return_value = 0
        mock_popen.return_value.returncode = 0

        executor = LocalPipelineExecutor()
        executor._parse_results_file = lambda *args, **kwargs: '1'
        executor.run_pipeline_collection(self.pipeline)

        self.assertEqual(len(executor.running_jobs), 0)
        self.assertFalse(mock_sleep.called)

    @mock.patch('os.makedirs')
    @mock.patch('os.chdir')
    @mock.patch('time.sleep')
    @mock.patch('subprocess.Popen')
    def test_jobs_are_polled_agained_when_poll_is_None(self, mock_popen, *args):
        polled = {'yes': False, 'calls': 0}
        def poll():
            polled['calls'] += 1
            if polled['yes']:
                mock_popen.return_value.returncode = 0
                return 0
            else:
                polled['yes'] = True
                return None

        mock_popen.return_value.poll = poll

        executor = LocalPipelineExecutor(run_serial=False)
        executor._parse_results_file = lambda *args, **kwargs: '1'
        executor.run_pipeline_collection(self.pipeline)

        # Called twice for first step and once for second.
        self.assertGreater(polled['calls'], 0)