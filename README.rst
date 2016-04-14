DOEpipeline
===========

This is yet another pipeline package. What distinguishes `DOEpipeline` is
that it enables pipeline optimization using methodologies from statistical
`Design of Experiments (DOE) <https://en.wikipedia.org/wiki/Design_of_experiments>`_

Pipeline-config
---------------

The pipeline is specified in a YAML config file. Example:

.. code-block:: yaml

    before_run:  # Optional setup.
        environment_variables:
            # Variables which will be set in the execution environment of
            # the pipeline run.
            MY_DIR: ~/my/dir/
        scripts:
            - ./a_script_to_run_before_starting.sh
            - python a_second_one.py

    design:
        # Design type is case insensitive (Todo: docs with allowed designs).
        type: BoxBehnken

        factors:
            # Factors are quantitiative unless specified otherwise. The factor
            # names probably works with spaces as well (Todo: test that).
            FactorA:
                # Minimum value.
                min: 0
                # Maximum value.
                max: 40
                # Low initial value.
                low_init: 0
                # High initial value.
                high_init: 20
            FactorB:
                min: 0
                max: 4
                low_init: 1
                high_init: 2
            FactorC:
                low_init: 140
                high_init: 160

        responses:  # One or more.
            ResponseA: maximize
            ResponseB: minimize

    # File where final results are dumped. One file per "experiment" will be
    # produced.
    results_file: my_results.txt

    working_directory: ~/my_work_directory

    pipeline:
        # Specifies order of jobs. These names must match the job-names below.
        - MyFirstJob
        - MySecondJob
        - MyThirdJob

    MyFirstJob:
        # The script can be multi-line and has access to environment
        # variables set above.
        script: >
            bash first_script_command.sh -o $MY_DIR
        factors:
            # The factors must match factors in the design.
            FactorA:
                # If script_option is given, the current factor value will be
                # added as a option to the job-script.
                script_option: --factor_a
            FactorB:

    MySecondJob:
        script: >
            bash my_second_script.sh --expressionWithFactors {% FactorC %}
        factors:
            FactorC:
                # If substitute is given (and true), the script's {% FactorC %}
                # will be replaced with the current factor value.
                substitute: true

    MyThirdJob:
        script: python make_output.py -o {% results_file %}


* `before_run`: Specifies pipeline setup.

    * `environment_variables`: Key value pairs of environment variable names and values to set prior to running pipeline.

* `pipeline`: List specifying order of jobs in pipeline.

* `JobName`: Step in pipeline. The job-name is used in `pipeline`-listing.

    * `script`: Command line script to execute. Has access to environment variables specified in `before_run`. There are two way to specify factors varied according to experimental design.

        * Templating: Factors with `substitute` equal to `true` can be substituted  using templating tags in scripts. An example template tag is `{% MyFactor %}` which will be substituted with the current value of factor `MyFactor`.

        * Script options: Factors with a specified `script_option` will be added as options to the end of the current script.

* `factors`: Lists  scripts parameters varied according to current experimental design.

    * `FactorName`: unique identifier of current factor.

        * `factor_name`: Name of factor in current MODDE-design.

        * `script_option`: Option passed to script if factor is passed to step as a script option.

        * `substitute`: If equal to `true` factor will be substituted into script using template tag of `FactorName`.