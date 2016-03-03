import abc


class BasePipelineExecutor:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def disconnect(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def execute_command(self, *args, **kwargs):
        pass