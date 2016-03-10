import logging
from contextlib import contextmanager
from doepipeline.executor.base import BasePipelineExecutor, PipelineRunFailed

log = logging.getLogger(__name__)


class ScreenExecutorMixin(BasePipelineExecutor):

    def run_in_screens(self, job_steps, experiment_index, env_variables):
        """ Run the collection of jobs in parallel using screens.

        Example job_steps:
        >>> job_steps = [
        ...     ['./script_1 --opt1 factor1_1', './script_1 --opt1 factor1_2'],
        ...     ['./script_2 --opt2 factor2_1', './script_2 --opt2 factor2_2']
        ... ]

        :param job_steps: List of step-wise scripts.
        :type job_steps: list[list]
        :param experiment_index: List of job-names.
        :type experiment_index: list[str]
        """
        log.info('Run jobs in parallel using screens')
        base_command = self.base_command
        if not self.base_command.endswith('&'):
            base_command += ' &'

        for job_name in experiment_index:
            log.debug('Setting up screen: {}'.format(job_name))
            self._make_screen(job_name)
            with self.screen(job_name):
                self._cd(job_name)
                if env_variables is not None:
                    self._set_env_variables(env_variables)

        # Run all scripts of each step in parallel using screens.
        # If a step fails, raise PipelineRunFailed.
        for i, step in enumerate(job_steps, start=1):
            for script, job_name in zip(step, experiment_index):
                # Prepare log and command.
                log_file = self.base_log.format(name=job_name, i=i)

                try:
                    command = base_command.format(script=script)
                except KeyError:
                    has_log = True
                    command = base_command.format(script=script,
                                                  logfile=log_file)
                else:
                    has_log = False

                # Execute script in screen.
                with self.screen(job_name):
                    if has_log:
                        self.execute_command('touch {log}'.format(log=log_file))
                    log.debug('Executes: {}'.format(command))
                    self.execute_command(command, watch=True, job_name=job_name)

            try:
                self._wait_until_current_jobs_are_finished()
            except PipelineRunFailed:
                raise

    @contextmanager
    def screen(self, screen_name):
        """ Context-manager to run commands within Linux-screens.

        :param screen_name: Name of screen to connect to.
        """
        self._reconnect_screen(screen_name)
        try:
            yield
        finally:
            self._disconnect_current_screen()

    def _disconnect_current_screen(self):
        self.execute_command('screen -d')

    def _reconnect_screen(self, name):
        self.execute_command('screen -r {}'.format(name))

    def _make_screen(self, name):
        self.execute_command('screen -S {}'.format(name))
        self._disconnect_current_screen()


class BatchExecutorMixin(BasePipelineExecutor):

    def run_batches(self, job_steps, experiment_index, env_variables):
        """ Run the collection of jobs in parallel using batch execution.

        Example job_steps:
        >>> job_steps = [
        ...     ['./script_1 --opt1 factor1_1', './script_1 --opt1 factor1_2'],
        ...     ['./script_2 --opt2 factor2_1', './script_2 --opt2 factor2_2']
        ... ]

        Given above example, the following batch-scripts will be run::

            cd [exp_name1] && nohup ./script_1 --opt1 factor1_1 > [exp_name1]_step_1.log 2>&1 && cd .. &
            cd [exp_name2] && nohup ./script_1 --opt1 factor1_2 > [exp_name2]_step_1.log 2>&1 && cd .. &
            wait

        And when finished::

            cd [exp_name1] && nohup ./script_2 --opt2 factor2_1 > [exp_name1]_step_2.log 2>&1 && cd .. &
            cd [exp_name2] && nohup ./script_2 --opt2 factor2_2 > [exp_name2]_step_2.log 2>&1 && cd .. &
            wait

        :param job_steps: List of step-wise scripts.
        :type job_steps: list[list]
        :param experiment_index: List of job-names.
        :type experiment_index: list[str]
        :param env_variables: Environment variables to set.
        :type env_variables: dict
        """
        log.info('Run batch jobs')

        if self.base_command.endswith(' &'):
            base = self.base_command[:-2]
        else:
            base = self.base_command

        if env_variables is not None:
            self._set_env_variables(env_variables)

        base_command= 'cd {job_dir} && {{log_script}}' + base + ' && cd .. &'

        for i, step in enumerate(job_steps, start=1):
            commands = list()
            for script, job_name in zip(step, experiment_index):
                # Prepare log and command.
                log_file = self.base_log.format(name=job_name, i=i)

                try:
                    command = base_command.format(job_dir=job_name,
                                                  script=script)
                except KeyError:
                    command = base_command.format(job_dir=job_name,
                                                  script=script,
                                                  logfile=log_file)
                    log_script = 'touch {logfile} && '.format(logfile=log_file)
                    command = command.format(log_script=log_script)
                else:
                    command = command.format(log_script='')

                commands.append(command)

            commands.append('wait')
            batch_command = ' '.join(commands)
            self.execute_command(batch_command, watch=True,
                                 job_name='batch_{}'.format(i))

            try:
                self._wait_until_current_jobs_are_finished()
            except PipelineRunFailed:
                raise