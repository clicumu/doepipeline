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
        os.chdir(os.path.dirname(__file__))
        self.work_dir = 'work_dir'
        try:
            os.mkdir(self.work_dir)
        except FileExistsError:
            shutil.rmtree(self.work_dir)
            os.mkdir(self.work_dir)

    def tearDown(self):
        os.chdir(os.path.dirname(__file__))

    def test_run(self):
        generator = PipelineGenerator.from_yaml('simple_test.yaml')

        designer = generator.new_designer_from_config()
        design = designer.new_design()
        pipeline = generator.new_pipeline_collection(design)

        cmd = '{script}'
        executor = LocalSerialExecutor(workdir=self.work_dir, base_command=cmd)
        results = executor.run_pipeline_collection(pipeline)
        optimum = designer.update_factors_from_response(results)
        pass