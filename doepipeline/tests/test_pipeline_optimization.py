import unittest
import os
import shutil
try:
    from unittest import mock
except ImportError:
    import mock
from doepipeline.generator import PipelineGenerator
from doepipeline.executor import LocalSerialExecutor


class TestLocalSerialRun(unittest.TestCase):

    def setUp(self):
        self.work_dir = 'work_dir'
        try:
            os.mkdir(self.work_dir)
        except FileExistsError:
            pass

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_run(self):
        generator = PipelineGenerator.from_yaml('simple_test.yaml')

        designer = generator.new_designer_from_config()
        design = designer.new_design()
        pipeline = generator.new_pipeline_collection(design)

        executor = LocalSerialExecutor(workdir=self.work_dir)
        results = executor.run_pipeline_collection(pipeline)
        pass