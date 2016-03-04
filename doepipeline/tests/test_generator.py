import copy
import unittest
import pandas as pd
import yaml
import os

from doepipeline.generator import PipelineGenerator


class BaseGeneratorTestCase(unittest.TestCase):

    def setUp(self):
        self.script_w_opt = {
            'script': './script_a',
            'factors': {
                'FactorA': {
                    'factor_name': 'Factor A',
                    'script_option': '--option'
                }
            }
        }

        self.script_w_sub = {
            'script': './script_b {% FactorB %}',
            'factors': {
                'FactorB': {
                    'factor_name': 'Factor B',
                    'substitute': True
                }
            }
        }

        self.config = {
            'pipeline': ['ScriptWithOptions', 'ScriptWithSub'],
            'ScriptWithOptions': self.script_w_opt,
            'ScriptWithSub': self.script_w_sub
        }
        self.yaml_path = 'test.yaml'
        f = open(self.yaml_path, 'w')
        f.write(yaml.dump(self.config))
        f.close()

        values = [
            ['A', .1, .2],
            ['B', .3, .4]
        ]
        self.design = pd.DataFrame(
                values, columns=['Exp Id', 'Factor A', 'Factor B']
        )

    def tearDown(self):
        os.remove(self.yaml_path)


class TestCreate(BaseGeneratorTestCase):

    def test_creating_doesnt_crash(self):
        generator = PipelineGenerator(self.config)

    def test_creating_from_yaml_doesnt_crash(self):
        yaml_generator = PipelineGenerator.from_yaml(self.yaml_path)
        with open(self.yaml_path) as f:
            buffer_generator = PipelineGenerator.from_yaml(f)

    def test_bad_yaml_raises_valueerror(self):
        self.assertRaises(ValueError,
                          lambda: PipelineGenerator.from_yaml({}))

    def test_bad_config_raises_valueerror(self):
        # Create bad pipeline by "skipping" one step in pipeline
        bad_config = copy.deepcopy(self.config)
        bad_config['pipeline'].pop()
        self.assertRaises(ValueError,
                          lambda: PipelineGenerator(bad_config))

class TestMakePipeline(BaseGeneratorTestCase):

    def setUp(self):
        super(TestMakePipeline, self).setUp()
        self.runner = PipelineGenerator(self.config)
        self.scripts1 = ['./script_a --option 0.1', './script_b 0.2']
        self.scripts2 = ['./script_a --option 0.3', './script_b 0.4']

    def test_render_experiments_without_id_column(self):
        pipeline_collection = self.runner.new_pipeline_collection(self.design)
        self.assertDictEqual({'0': self.scripts1, '1': self.scripts2}, pipeline_collection)

    def test_render_experiments_with_id_column(self):
        new_collection = self.runner.new_pipeline_collection(self.design, 'Exp Id')
        self.assertDictEqual({'A': self.scripts1, 'B': self.scripts2}, new_collection)
