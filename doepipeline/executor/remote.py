"""
This module contains executors for pipeline execution remotely.
"""
import paramiko
import socket
import logging
import time
import re
from contextlib import contextmanager
from doepipeline.executor.base import BasePipelineExecutor, CommandError
from doepipeline.executor.mixins import ScreenExecutorMixin, BatchExecutorMixin

log = logging.getLogger(__name__)


class ConnectionFailed(Exception):
    pass


class BaseSSHExecutor(BasePipelineExecutor):

    """

    """

    def __init__(self, credentials, reconnect=True,
                 connect_timeout=10, connection_attempts=10, *args, **kwargs):
        try:
            assert isinstance(credentials, dict),\
                'credentials must be a dictionary SSH-host credentials'
            assert 'host' in credentials,\
                'credentials must at least contain a host'
            assert isinstance(reconnect, bool),\
                'reconnect must be bool'
            assert isinstance(connect_timeout, (float, int)) and connect_timeout > 0,\
                'connect_timeout must be positive number'
            assert isinstance(connection_attempts, int) and connection_attempts > 0,\
                'connection_attempts must be positive integer'
        except AssertionError as e:
            raise ValueError(str(e))

        super(BaseSSHExecutor, self).__init__(*args, **kwargs)
        self._client = None
        self.host = credentials.get('host')
        self.reconnect = reconnect
        self.credentials = {key: value for key, value in credentials.items()
                            if key != 'host'}
        self.connect_timeout = connect_timeout
        self.connection_attempts = connection_attempts

    def connect(self, connect_timeout=None, n_attempts=None):
        """ Connect to host using SSH.

        All keyword arguments are passed to :py:method:`paramiko.SSHClient.connect`.
        See `documentation <http://docs.paramiko.org/en/1.16/api/client.html#paramiko.client.SSHClient.connect>`_

        :param int connect_timeout: Seconds to time-out between attempts.
        :param int n_attempts: Maximum number of attempts.
        :raises ConnectionFailed:
            If run out of number of connection attempts
            or authentication failed.
        """
        connect_timeout = connect_timeout if connect_timeout is not None \
            else self.connect_timeout
        n_attempts = n_attempts if n_attempts is not None\
            else self.connection_attempts
        try:
            assert isinstance(n_attempts, int) and n_attempts > 0,\
                'n_attempts must be positive integer'
            assert isinstance(connect_timeout, (int, float)) \
                   and connect_timeout > 0,\
                'connect_timeout must be positive number'
        except AssertionError as e:
            raise ValueError(str(e))

        for attempt in range(n_attempts):
            try:
                log.info('Attempts to connect to: {host}'.format(host=self.host))
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.load_system_host_keys()
                client.connect(self.host, **self.credentials)
            except (paramiko.BadHostKeyException,
                    paramiko.AuthenticationException) as e:
                # If authentication fails or host key is wrong, fail
                # immediately.
                log.critical('Connection failed: ' + str(e))
                raise ConnectionFailed(str(e))
            except (paramiko.SSHException, socket.error):
                # If SSH-connection fails, try to connect again.
                msg = 'Connection attempt {i} failed, retrying...'
                log.warn(msg.format(i=attempt + 1))
                time.sleep(connect_timeout)
            else:
                log.info('Connection succeeded')
                self._client = client
                return

        log.critical('Connection failed, run out of attempts.')
        self._client = None
        raise ConnectionFailed

    def disconnect(self):
        """ Disconnect from host. """
        log.info('Disconnecting')
        self._client.close()
        self._client = None

    def execute_command(self, command, watch=False, **kwargs):
        """ Execute script over SSH.

        :param str command: Command to execute.
        :param bool watch: If True, monitor process.
        :param kwargs: Keyword arguments-
        :return: stdin, stdout, stderr
        :rtype: tuple[str]
        """
        super(BaseSSHExecutor, self).execute_command(command, watch, **kwargs)
        if watch:
            try:
                assert command.startswith('nohup'),\
                    'If watch, command must be nohupped'
                assert command.endswith('echo $!'),\
                    'If watch, pid must be echoed using "echo $!"'
                # Crude matching for redirect of stdout. Probably should be
                # more specific since it matches "2>&1" as well as
                # "script > logfile" or "script >/dev/null".
                assert re.search(r'>\s?\S*', command) is not None,\
                    ('If watch, stdout must be redirected. '
                     'pid-check hangs otherwise.')
            except AssertionError as e:
                raise ValueError(str(e))

        try:
            with self.connection(disconnect=watch):
                stdin, stdout, stderr = self._client.exec_command(command)
        except paramiko.SSHException as e:
            raise CommandError(str(e))
        else:
            if watch:
                pids = [line.strip() for line in stdout.readlines()]
                if len(pids) > 1:
                    self.running_jobs = zip(kwargs['job_name'], pids)
                else:
                    self.running_jobs[kwargs['job_name']] = pids[0]

        return stdin, stdout, stderr

    @contextmanager
    def connection(self, disconnect=True):
        """ Context-manager which reconnects when not connected.

        :param bool disconnect: If True, disconnect after call.
        """
        transport = self._client.get_transport() \
            if self._client is not None else None

        if transport is None or not transport.is_active():
            try:
                self.connect()
            except ConnectionFailed:
                raise
            else:
                log.info('Reconnection succeeded.')
        try:
            yield
        finally:
            if disconnect:
                self.disconnect()


class SSHScreenExecutor(ScreenExecutorMixin, BaseSSHExecutor):
    """
    Executor class which executes jobs in parallel using screens
    at remote host communicating via SSH.
    """

    def poll_jobs(self):
        with self.connection():
            ScreenExecutorMixin.poll_jobs(self)


class SSHBatchExecutor(BatchExecutorMixin, BaseSSHExecutor):
    """
    Executor class which executes jobs in parallel using batch
    scripts at remote host communicating via SSH.
    """

    def poll_jobs(self):
        with self.connection():
            BatchExecutorMixin.poll_jobs(self)