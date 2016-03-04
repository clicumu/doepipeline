import copy
import unittest
import pandas as pd
import yaml
import os

from doepipeline.generator import PipelineGenerator


class TestPipelineRunner(unittest.TestCase):

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

    def test_create_runner(self):
        # Assert no crash
        generator = PipelineGenerator(self.config)
        yaml_generator = PipelineGenerator.from_yaml(self.yaml_path)
        with open(self.yaml_path) as f:
            buffer_generator = PipelineGenerator.from_yaml(f)

        self.assertRaises(ValueError,
                          lambda: PipelineGenerator.from_yaml({}))

        # Create bad pipeline by "skipping" one step in pipeline
        bad_config = copy.deepcopy(self.config)
        bad_config['pipeline'].pop()
        self.assertRaises(ValueError,
                          lambda: PipelineGenerator(bad_config))

    def test_render_experiments(self):
        runner = PipelineGenerator(self.config)

        pipeline_collection = runner.new_pipeline_collection(self.design)
        scripts1 = ['./script_a --option 0.1', './script_b 0.2']
        scripts2 = ['./script_a --option 0.3', './script_b 0.4']
        self.assertDictEqual({'0': scripts1, '1': scripts2}, pipeline_collection)

        new_collection = runner.new_pipeline_collection(self.design, 'Exp Id')
        self.assertDictEqual({'A': scripts1, 'B': scripts2}, new_collection)
