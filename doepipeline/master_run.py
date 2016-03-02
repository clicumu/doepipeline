from pymoddeq.runner import ModdeQRunner
from pipeline_runner import PipelineRunner
import os

YAML_SETTINGS_FILE = "investigation.yaml"
YAML_PIPELINE = "pipeline.yaml"


# Set up the modde investigation based on the user specified yaml file
modde = ModdeQRunner.from_yaml(YAML_SETTINGS_FILE, overwrite_mip=True)

# Set up the PipelineRunner that will handle running experiments (user made code)
pipeline_runner = PipelineRunner(YAML_PIPELINE)


def optimization_loop(max_iterations):
    done = False
    iteration = 0
    print "Initiating optimization loop..."  # tmp
    while not done and iteration < max_iterations:
        # New iteration
        iteration += 1
        print "Iteration: %i" % iteration  # tmp
        pipeline_runner.next_iteration()  # PipelineRunner next iteration
        print "Retrieving the experimental setup..."  # tmp
        exp_setup = modde.get_experimental_setup()  # Get the experimental setup from MODDE (the worksheet with experiments)
        print exp_setup  # tmp
        print "Set the experimental design..."  # tmp
        pipeline_runner.set_exp_design(exp_design=exp_setup)  # Set the experimental setup
        print "Constructing the pipeline..."  # tmp
        pipeline_runner.new_pipeline()  # Write a pipeline based on the experimental setup
        print "Running the pipeline..."  # tmp
        pipeline_runner.run_pipeline()  # Run the pipeline

        # Get the result (user made code)
        print "Collecting the raw result..."  # tmp
        raw_result = pipeline_runner.collect_raw_result(pipeline_runner.yaml_dict['inv_name'] + '_vcf_compare.txt')  # The raw result
        print "Calculating the result..."  #tmp
        worksheet_complete = pipeline_runner.calculate_result(raw_result, exp_setup)  # Insert the response values into the Pandas DataFrame

        # Save a local copy of the updated Pandas DataFrame
        print "Saving a local copy of the updated pandas dataframe..."  # tmp
        worksheet_complete.to_csv(
            os.path.join(
                modde.settings['investigationFolder'],
                "iter_%i.txt" % pipeline_runner.iteration
            ),
            sep="\t"
        )

        # Insert the completed worksheet, perform optimization and return a new experimental setup based on an updated design space
        print "Inserting completed worksheet into MODDE, optimizing and moving design space..."  # tmp
        exp_setup = modde.optimize_and_move_design_space(worksheet_complete)  # will return None when the design space wasn't updated

        if exp_setup is None:
            print "The design space was not moved, exiting..."  # tmp
            done = True

        print "The design space was moved, reiterating..."  # tmp

optimization_loop(max_iterations=10)