try:
    from unittest import mock
except ImportError:
    import mock
import socket
import paramiko

from doepipeline.executor.remote import BaseSSHExecutor, ConnectionFailed
from doepipeline.executor.base import CommandError
from doepipeline.tests.executor_utils import ExecutorTestCase


class MockSSHExecutor(BaseSSHExecutor):

    def poll_jobs(self):
        pass

    def run_jobs(self, job_steps, experiment_index, env_variables):
        pass

    def read_file_contents(self, filename):
        return 'A,1\nB,2'


class BaseSSHTestCase(ExecutorTestCase):

    credentials = {
        'host': 'localhost',
        'port': 9999,
        'username': 'testadmin',
        'password': 'x'
    }
    host = credentials['host']
    connect_kwargs = {key: value for key, value in credentials.items()
                      if key != 'host'}

    executor_class = MockSSHExecutor
    init_args = (credentials, )

    @mock.patch('paramiko.SSHClient')
    def test_execute_commands_returns_tuple(self, mock_client):
        mock_client.return_value.exec_command.return_value = 'in', 'out', 'err'
        super(BaseSSHTestCase, self).test_execute_commands_returns_tuple()


class TestSSHExecutorSetup(BaseSSHTestCase):

    def test_setup_doesnt_crash_given_credentials(self):
        executor = MockSSHExecutor(self.credentials)

    def test_setup_crash_given_bad_arguments(self):
        bad_kwargs = [
            {'credentials': {}},
            {'credentials': None},
            {'credentials': {'user': 'some'}},
            {'credentials': {'host': 'some'}, 'reconnect': 'true'},
            {'credentials': {'host': 'some'}, 'reconnect': 1},
            {'credentials': {'host': 'some'}, 'reconnect': [1]},
            {'credentials': {'host': 'some'}, 'reconnect': None},
            {'credentials': {'host': 'some'}, 'connection_attempts': 1.0},
            {'credentials': {'host': 'some'}, 'connection_attempts': '1'},
            {'credentials': {'host': 'some'}, 'connection_attempts': -1},
            {'credentials': {'host': 'some'}, 'connection_attempts': None},
            {'credentials': {'host': 'some'}, 'connect_timeout': '1'},
            {'credentials': {'host': 'some'}, 'connect_timeout': -1},
            {'credentials': {'host': 'some'}, 'connect_timeout': None},
        ]

        for kwargs in bad_kwargs:
            self.assertRaises(ValueError, MockSSHExecutor, **kwargs)


class TestSSHExecutorConnect(BaseSSHTestCase):

    @mock.patch('paramiko.SSHClient.connect')
    def test_connect_doesnt_crash(self, mock_connect):
        executor = MockSSHExecutor(self.credentials)
        executor.connect()

        mock_connect.assert_called_with(self.host, **self.connect_kwargs)


    @mock.patch('paramiko.SSHClient.connect')
    def test_connect_fails_with_bad_values(self, mock_connect):
        executor = MockSSHExecutor(self.credentials)

        bad_kwargs = [
            {'n_attempts': 1.0},
            {'n_attempts': '1'},
            {'n_attempts': -1},
            {'connect_timeout': '1'},
            {'connect_timeout': -1},
            {'connect_timeout': 2j}
        ]
        for kwarg in bad_kwargs:
            self.assertRaises(ValueError, executor.connect, **kwarg)
            self.assertFalse(mock_connect.called)


    @mock.patch('time.sleep')
    @mock.patch('paramiko.SSHClient.connect')
    def test_auth_errors_raises_ConnectionFailed(self, mock_connect, mock_sleep):
        error = paramiko.AuthenticationException(mock.Mock(), 'auth error')
        mock_connect.side_effect = error
        executor = MockSSHExecutor(self.credentials)

        self.assertRaises(ConnectionFailed, executor.connect)
        self.assertFalse(mock_sleep.called)

        def new_error(*args, **kwargs):
            raise paramiko.BadHostKeyException('host', paramiko.PKey(),
                                               paramiko.PKey())

        mock_connect.side_effect = new_error

        self.assertRaises(ConnectionFailed, executor.connect)
        self.assertFalse(mock_sleep.called)


class TestSSHExecutorReconnect(BaseSSHTestCase):

    @mock.patch('time.sleep')
    @mock.patch('paramiko.SSHClient.connect')
    def test_failed_connection_reconnects_on_socketerror(self, mock_connect, mock_sleep):
        called = {'yes': False}

        def side_effect(*args, **kwargs):
            if not called['yes']:
                called['yes'] = True
                raise socket.error
            else:
                return

        mock_connect.side_effect = side_effect
        executor = MockSSHExecutor(self.credentials)
        executor.connect()
        self.assertEqual(mock_sleep.call_count, 1)

    @mock.patch('time.sleep')
    @mock.patch('paramiko.SSHClient.connect')
    def test_failed_connection_reconnects_on_SSHException(self, mock_connect,
                                                          mock_sleep):
        called = {'yes': False}
        def side_effect(*args, **kwargs):
            if not called['yes']:
                called['yes'] = True
                raise paramiko.SSHException
            else:
                return

        mock_connect.side_effect = side_effect
        executor = MockSSHExecutor(self.credentials)
        executor.connect()
        self.assertEqual(mock_sleep.call_count, 1)

    @mock.patch('time.sleep')
    @mock.patch('paramiko.SSHClient.connect')
    def test_failed_reconnects_raises_ConnectionFailed(self, mock_connect, mock_sleep):
        mock_connect.side_effect = socket.error(mock.Mock(), 'socket error')
        n = 10
        executor = MockSSHExecutor(self.credentials, connection_attempts=n)
        self.assertRaises(ConnectionFailed, executor.connect)
        self.assertEqual(mock_sleep.call_count, n)

        mock_connect.side_effect = paramiko.SSHException(mock.Mock(), 'SSH exception')
        self.assertRaises(ConnectionFailed, executor.connect)
        self.assertEqual(mock_sleep.call_count, 2 * n)

    @mock.patch('paramiko.SSHClient')
    def test_execution_connects_if_disconnected(self, mock_client):
        mock_client.return_value.exec_command.return_value = 'in', 'out', 'err'
        executor = MockSSHExecutor(self.credentials, reconnect=False)

        def set_client():
            executor._client = mock_client()

        executor.connect = mock.MagicMock()
        executor.connect.side_effect = set_client
        self.assertFalse(executor.connect.called)
        executor.execute_command('ls')
        self.assertTrue(executor.connect.called)

    @mock.patch('paramiko.SSHClient.connect')
    def test_execution_autoconnect_raises_ConnectionFailed_on_fail(self, mock_connect):
        executor = MockSSHExecutor(self.credentials)
        error = paramiko.AuthenticationException(mock.Mock(), 'auth error')
        mock_connect.side_effect = error

        self.assertRaises(ConnectionFailed, executor.execute_command, 'ls')


class TestSSHExecutorDisconnect(BaseSSHTestCase):

    @mock.patch('paramiko.SSHClient.connect')
    @mock.patch('paramiko.SSHClient.close')
    def test_disconnect_is_called(self, mock_close, mock_connect):
        executor = MockSSHExecutor(self.credentials)
        executor.connect()

        executor.disconnect()
        self.assertTrue(mock_close.called)

    @mock.patch('paramiko.SSHClient')
    def test_disconnect_after_call_if_set_to_watch(self, mock_client):
        mock_file = mock.MagicMock()
        mock_file.readlines.return_value = ['jobname']
        mock_client.return_value.exec_command.return_value = 'in', mock_file, 'err'
        executor = MockSSHExecutor(self.credentials)
        executor.disconnect = mock.MagicMock()

        self.assertFalse(executor.disconnect.called)
        executor.execute_command('nohup ls >/dev/null & echo $!', watch=True, job_name='test')
        self.assertTrue(executor.disconnect.called)

    @mock.patch('paramiko.SSHClient')
    def test_disconnect_is_not_called_when_not_watching(self, mock_client):
        mock_client.return_value.exec_command.return_value = 'in', 'out', 'err'
        executor = MockSSHExecutor(self.credentials)
        executor.disconnect = mock.MagicMock()

        self.assertFalse(executor.disconnect.called)
        executor.execute_command('ls')
        self.assertFalse(executor.disconnect.called)


class TestSSHExecutorExecute(BaseSSHTestCase):

    @mock.patch('paramiko.SSHClient')
    def test_raises_CommandError_on_failed_ssh(self, mock_client):
        error = paramiko.SSHException(mock.Mock())
        mock_client.return_value.exec_command.side_effect = error

        executor = MockSSHExecutor(self.credentials)
        self.assertRaises(CommandError, executor.execute_command, 'ls')