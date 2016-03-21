from doepipeline.generator import PipelineGenerator
from doepipeline.executor import LocalPipelineExecutor
from doepipeline.designer import BaseExperimentDesigner
import pandas as pd
import numpy as np


class ExampleDesigner(BaseExperimentDesigner):

    def __init__(self, *args, **kwargs):
        super(ExampleDesigner, self).__init__(*args, **kwargs)
        self.design = None
        np.random.seed(123456789)

    def new_design_from_response(self, response):
        self.design += 1
        return self.design

    def new_design(self, factor_settings=None):
        data = np.random.randint(10, size=(2, len(self.factors)))
        self.design = pd.DataFrame(data, index=['A', 'B'],
                                   columns=self.factors.keys())
        return self.design


if __name__ == '__main__':
    generator = PipelineGenerator.from_yaml('example_pipeline.yaml')
    designer = generator.new_designer_from_config(ExampleDesigner)
    design = designer.new_design()
    pipeline = generator.new_pipeline_collection(design)
    executor = LocalPipelineExecutor()
    results = executor.run_pipeline_collection(pipeline)
    design = designer.new_design_from_response(results)