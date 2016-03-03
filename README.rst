DOEpipeline
===========

This is yet another pipeline package. What distinguishes `DOEpipeline` is
that it enables pipeline optimization using methodologies from statistical
`Design of Experiments (DOE) <https://en.wikipedia.org/wiki/Design_of_experiments>`_

Pipeline-config
---------------

The pipeline is specified in a YAML config file. Example:

.. code-block:: yaml

    before_run:
        environment_variabels:
            MY_DIR: ~/my/dir/

    pipeline:
        - MyFirstJob
        - MySecondJob

    MyFirstJob:
        script: >
            ./first_script_command -o $MY_DIR
        factors:
            FactorA:
                factor_name: Factor A
                script_option: --factor_a
            FactorB:
                factor_name: Factor B
                script_option: --factor_b

    MySecondJob:
        script: >
            ./my_second_script --expressionWithFactors {% FactorC %}
        factors:
            FactorC:
                factor_name: Factor C
                substitute: true


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