import types
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

    def test_run_pipeline_sets_up_properly(self):

        for workdir in (None, 'foldername', 'path/to/folder'):
            executor = MockBaseExecutor(workdir=workdir)

            # Monkey patch out screen-run to truncate script output.
            executor.run_in_screens = lambda steps, index, envs: None

            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = [
                'cd {}'.format(workdir if workdir is not None else '.'),
                'mkdir {}'.format(self.design['Exp Id'][0]),
                'mkdir {}'.format(self.design['Exp Id'][1])
            ]
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_run_pipeline_sets_up_properly_when_no_workdir(self):
        for workdir in (None, 'foldername', 'path/to/folder'):
            executor = MockBaseExecutor(workdir=workdir)

            # Monkey patch out screen-run to truncate script output.
            executor.run_in_screens = lambda steps, index, envs: None

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
        executor = MockScreenExecutor()

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
        self.assertListEqual(expected_steps, output['steps'])


class TestBaseExecutorExecutions(ExecutorTestCase):

    def test_bad_command_input_raises_ValueError(self):
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


class TestBaseExecutorRunsScreens(ExecutorTestCase):

    def make_expected_scripts(self, workdir, base_script, executor,
                              use_log=False, system='Windows'):
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
                'cd {}'.format(job)
            ]
            expected_scripts += [
                '{}={}'.format(key, value)\
                for key, value in self.env_vars.items()
            ]
            expected_scripts.append('screen -d')

        # Screen script runs.
        for step in range(2):
            for job, i in zip((job1, job2), (1, 2)):
                expected_scripts += ['screen -r {}'.format(job)]

                if use_log:
                    log_file = executor.base_log.format(name=job, i=step + 1)

                    if system == 'Windows':
                        expected_scripts += ['type NUL >> {}'.format(log_file)]
                    else:
                        expected_scripts += ['touch {}'.format(log_file)]
                    script = base_script.format(script=self.pipeline[job][step],
                                                logfile=log_file)
                else:
                    script = base_script.format(script=self.pipeline[job][step])

                script += ' & echo $!'
                expected_scripts += [
                    script,
                    'screen -d'
                ]
        return expected_scripts

    def test_run_jobs_without_logs_give_correct_output(self):
        base_cmd = '{script}'
        executor = MockScreenExecutor(base_command=base_cmd)
        executor.run_pipeline_collection(self.pipeline)
        expected_scripts = self.make_expected_scripts('.', base_cmd,
                                                      executor, False)
        self.assertListEqual(expected_scripts, executor.scripts)

    def test_run_jobs_with_logs_give_corrext_output(self):
        base_cmd = '{script}_{logfile}'
        executor = MockScreenExecutor(base_command=base_cmd)
        executor.run_pipeline_collection(self.pipeline)
        expected_scripts = self.make_expected_scripts('.', base_cmd,
                                                      executor, True)
        self.assertListEqual(expected_scripts, executor.scripts)

    def test_failed_screen_run_poll_raises_PipelineRunFailed(self):
        executor = MockScreenExecutor(base_command='{script}')
        def poll_fail():
            return MockBaseExecutor.JOB_FAILED, 'Error'

        executor.poll_jobs = poll_fail

        with self.assertRaises(PipelineRunFailed) as cm:
            executor.run_pipeline_collection(self.pipeline)
            self.assertEqual(cm.message, 'Error')

    @mock.patch('time.sleep')
    def test_running_screens_polls_again_and_breaks_when_finished(self,
                                                                  mock_sleep):
        executor = MockScreenExecutor(base_command='{script}', poll_interval=1)

        poll_checks = {'polls': 0, 'checked': False}
        def mock_poll():
            poll_checks['polls'] += 1
            if poll_checks['checked']:
                return MockBaseExecutor.JOB_FINISHED, ''
            else:
                poll_checks['checked'] = True
                return MockBaseExecutor.JOB_RUNNING, ''

        executor.poll_jobs = mock_poll
        executor.run_pipeline_collection(self.pipeline)
        self.assertGreaterEqual(poll_checks['polls'], 2)
        mock_sleep.assert_called_with(executor.poll_interval)


class TestBaseExecutorRunsBatches(ExecutorTestCase):

    def make_expected_scripts(self, workdir, base_script, executor, use_log=False):
        job1, job2 = self.design['Exp Id'].tolist()

        # Run-setup.
        expected_scripts = ['cd {}'.format(workdir)]
        expected_scripts += [
            'mkdir {}'.format(job1),
            'mkdir {}'.format(job2)
        ]
        expected_scripts += [
            '{}={}'.format(key, value)
            for key, value in self.env_vars.items()
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
            executor = MockBatchExecutor(base_command=base_script)
            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = self.make_expected_scripts('.', '{script}', executor)
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_run_batch_with_log_gives_correct_output(self):
        base_command = '{script} > {logfile}'
        for command in (base_command, base_command + ' &'):
            executor = MockBatchExecutor(base_command=command)
            executor.run_pipeline_collection(self.pipeline)
            expected_scripts = self.make_expected_scripts('.', base_command,
                                                          executor, use_log=True)
            self.maxDiff = None
            self.assertListEqual(expected_scripts, executor.scripts)

    def test_failed_batch_run_poll_raises_PipelineRunFailed(self):
        executor = MockBatchExecutor(base_command='{script}')
        def poll_fail():
            return MockBaseExecutor.JOB_FAILED, 'Error'

        executor.poll_jobs = poll_fail

        with self.assertRaises(PipelineRunFailed) as cm:
            executor.run_pipeline_collection(self.pipeline)
            self.assertEqual(cm.message, 'Error')

    @mock.patch('time.sleep')
    def test_when_running_polls_again_and_breaks_when_finished(self,
                                                                  mock_sleep):
        executor = MockBatchExecutor(base_command='{script}')

        poll_checks = {'polls': 0, 'checked': False}
        def mock_poll():
            poll_checks['polls'] += 1
            if poll_checks['checked']:
                return MockBaseExecutor.JOB_FINISHED, ''
            else:
                poll_checks['checked'] = True
                return MockBaseExecutor.JOB_RUNNING, ''

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
