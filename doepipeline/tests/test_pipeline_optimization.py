import unittest
import os
import shutil
import numpy as np
import yaml
try:
    from unittest import mock
except ImportError:
    import mock
from doepipeline.generator import PipelineGenerator
from doepipeline.executor import LocalPipelineExecutor


class BaseRunTestCase(unittest.TestCase):

    executor = LocalPipelineExecutor

    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        self.work_dir = os.path.join(os.getcwd(), 'work_dir')
        with open('simple_test.yaml') as f:
            self.config = yaml.load(f)
            self.config['working_directory'] = self.work_dir

        try:
            os.mkdir(self.work_dir)
        except FileExistsError:
            shutil.rmtree(self.work_dir)
            os.mkdir(self.work_dir)

    def tearDown(self):
        os.chdir(os.path.dirname(__file__))

    def config_with_optimum(self, x, y):
        config = self.config.copy()
        config['constants'] = {'ROOT': os.getcwd()}
        script = 'python {% ROOT make_output.py %}'
        script += ' -x %d -y %d' %(x, y)
        script += ' step_one.txt step_two.txt -o {% results_file %}'
        config['MyThirdJob']['script'] = script
        return config


class TestLocalSerialRun(BaseRunTestCase):

    def test_run_optimum_within_bounds(self):
        generator = PipelineGenerator(self.config_with_optimum(15, 3))

        designer = generator.new_designer_from_config()
        design = designer.new_design()
        pipeline = generator.new_pipeline_collection(design)

        cmd = '{script}'
        executor = self.__class__.executor(workdir=self.work_dir, base_command=cmd)
        results = executor.run_pipeline_collection(pipeline)
        optimum = designer.update_factors_from_response(results)

        expected_optimum = {'FactorA': 15, 'FactorB': 3}
        self.assertTrue(optimum.converged)
        for factor in design.columns:
            self.assertTrue(np.isclose(optimum.predicted_optimum[factor],
                                       expected_optimum[factor]))

    def test_run_optimum_outside_bounds(self):
        generator = PipelineGenerator(self.config_with_optimum(5, 5))

        designer = generator.new_designer_from_config()
        design = designer.new_design()
        pipeline = generator.new_pipeline_collection(design)

        cmd = '{script}'
        executor = self.__class__.executor(workdir=self.work_dir, base_command=cmd)
        results = executor.run_pipeline_collection(pipeline)
        optimum = designer.update_factors_from_response(results)

        expected_optimum = {'FactorA': 5, 'FactorB': 5}
        self.assertFalse(optimum.converged)
        for factor in design.columns:
            self.assertTrue(np.isclose(optimum.predicted_optimum[factor],
                                       expected_optimum[factor]))


class TestLocalParallelRun(TestLocalSerialRun):

    executor = lambda *a, **kw: LocalPipelineExecutor(*a, run_serial=False,
                                                      poll_interval=1, **kw)


class TestLocalSerialScreeningRun(BaseRunTestCase):

    def setUp(self):
        super(TestLocalSerialScreeningRun, self).setUp()
        self.config['design']['factors']['FactorB']['min'] = 0
        self.config['design']['factors']['FactorB']['max'] = 50

    def test_screening_finds_optimum_at_origin(self):
        generator = PipelineGenerator(self.config_with_optimum(0, 0))
        designer = generator.new_designer_from_config(skip_screening=False)
        design = designer.new_design()
        pipeline = generator.new_pipeline_collection(design)

        cmd = '{script}'
        executor = self.__class__.executor(workdir=self.work_dir,
                                           base_command=cmd)
        results = executor.run_pipeline_collection(pipeline)
        optimum = designer.update_factors_from_response(results)

        self.assertEqual(designer._phase, 'optimization')
        self.assertFalse(optimum.converged)
        expected_optimum = {'FactorA': 0, 'FactorB': 0}
        for factor in design.columns:
            self.assertTrue(np.isclose(optimum.predicted_optimum[factor],
                                       expected_optimum[factor]))

    def test_optimization_founds_optimum_after_screening(self):
        generator = PipelineGenerator(self.config_with_optimum(15, 3))
        designer = generator.new_designer_from_config(skip_screening=False)
        cmd = '{script}'
        executor = self.__class__.executor(workdir=self.work_dir,
                                           base_command=cmd)
        for i in range(3):
            design = designer.new_design()
            pipeline = generator.new_pipeline_collection(design)
            results = executor.run_pipeline_collection(pipeline)
            optimum = designer.update_factors_from_response(results)

        expected_optimum = {'FactorA': 15, 'FactorB': 3}
        self.assertTrue(optimum.converged)
        for factor in design.columns:
            self.assertTrue(np.isclose(optimum.predicted_optimum[factor],
                                       expected_optimum[factor]))


class TestLocalParallelScreeningRun(TestLocalSerialScreeningRun):

    executor = lambda *a, **kw: LocalPipelineExecutor(*a, run_serial=False,
                                                      poll_interval=1, **kw)

