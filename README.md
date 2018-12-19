doepipeline
===========

This is yet another pipeline package. What distinguishes `doepipeline` is
that it enables pipeline optimization using methodologies from statistical
`Design of Experiments (DOE) <https://en.wikipedia.org/wiki/Design_of_experiments>`_

Pipeline-config
---------------

The pipeline is specified in a YAML config file. Example:


    design:
        # Design type is case insensitive
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
            ResponseA: 
                criterion: maximize

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
        # The script can be multi-line.
        script: >
            bash first_script_command.sh {% FactorB %} -o $MY_DIR
        factors:
            # The factors must match factors in the design.
            FactorA:
                # If script_option is given, the current factor value will be
                # added as a option to the job-script.
                script_option: --factor_a
            FactorB:
                # If substitute is given (and true), the script's {% FactorC %}
                # will be replaced with the current factor value.
                substitute: true

    MySecondJob:
        script: >
            bash my_second_script.sh --expressionWithFactors {% FactorC %}
        factors:
            FactorC:
                substitute: true

    MyThirdJob:
        script: python make_output.py -o {% results_file %}
        
    before_run:  # Optional setup.
        environment_variables:
            # Variables which will be set in the execution environment of
            # the pipeline run.
            MY_DIR: ~/my/dir/
        scripts:
            - ./a_script_to_run_before_starting.sh
            - python a_second_one.py


The key-value pairs are specified below.
### `design` 

Specifies factors and responses to investigate as well as what design to use in key-value mapping. Valid keys specifying design are:

* `type`: Required. Design to use. Available choices (case-insensitive) are:
    * `ccc`/ `ccf`/ `cci`: Central composite designs.
    * `fullfactorial2levels`/`fullfactorial3levels`
    * `placketburman`
    * `boxbehnken`
* `factors`: Required. Mapping of one or more factors.
    * `<factor-name>`: Keys are name used for factor and will be used for substitutions. Values are specified below.
* `responses`: Required. Mapping of one or more response.
    * `<response-name>`: Keys are name used for response, values are specified below.
* `screening_reduction`: Optional. `auto` (default) or positive integer. Specifies reduction factors used for GSD during screening. 

### `<factor-name>`
Specification of each factor. Valid keys specifying factors are:
* `type` Optional. Type will be quantitative if not specified. Valid values are:
    * `quantitative`: Default. Numeric factor which can take real values.
    * `ordinal`: Numeric factor constrained to integer values.
    * `categorical`: Categorical values.
* `low_init`: Required for numeric factors. Low starting value.
* `high_init`: Required for numeric factors. High starting value.
* `max`: Optional for numeric factors. Maximum allowed value, default is positive infinity.
* `min`: Optional for numeric factors. Minimum allowed value, default is negative infinity.
* `values`: Required for categorical factors. Possible values.
* `screening_levels`: Optional for numeric factors. Number of levels investigated during screening phase. Default is 5.
### `<response-name>`
Specification of a response. Valid keys specifying responses are:

* `criterion`: Required for all responses. Valid values are:
    * `maximize` / `minimize`: Response will be maximized or minimized respectively.
    * `target`: Reach target value (optimum is neither above nor below value)
* `target`: Required when there are multiple responses for responses for any criterion.
* `low_limit`: Required when there are multiple responses for responses with criterion `target`or `maximize`, optional otherwise. Indicates lowest acceptable value.
* `high_limit`: Required when there are multiple responses for responses with criterion `target` or `minimize`, optional otherwise. Indicates highest acceptable value.
* `transform`: Optional. Indicates what transform used prior to optimization: Valid values are:
    * `log`: Log transformation
    * `box-cox`: [Box-Cox transformation](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation).

### `pipeline`
Required. Ordered list specifying order of jobs. Values are `<job-name>` specified below.

### `<job-name>`

Specification of each job that will be run in the order specified in `pipeline`. Optionally, factors will be substituted.

* `script`: Required. String containing command that will be executed. May use substitution (see below) to parameterize commands. Substitution or appending of script option is required to input factors currently optimized.
* `factors`: Optional. Contains mapping of what factors that should be used in the current experiment. Valid keys are:
    * `<factor-name>`: One of the factors specified in `design`. Each factor must carry one of the two following key-value pairs to indicate how it should be input in the job:
        * `substitute: true`: Factor will be substituted using templating.
        * `script_option: --<option-name>`: A option flag will be appended to the script string. E.g., if `script_option: -a` is provided for factor `FactorA`, then `-a <value-of-FactorA>` will be appended to the script string.

### `results_file`
Required. Indicates the name of the file containing the results from each pipeline run, may be used for substitution. A results-file for each factor setup will be produced in the working-directory for the current experiment.

### `working_directory`
Required. Root directory which will contain the results from all iterations and experiments.

### `before_run`
Optional. Specify setup. Valid keys are:
* `environment_variables`: Key-value pairs of environment variables to be set prior to running pipeline.
* `scripts`: Ordered list of scripts to run prior to running pipeline.

### `constants`
Optional. Key-value pairs of constants accessible for substitution. Keys written in upper-case are interpreted as directories and
can be used for path-substitution.

### Substitution
`doepipeline` uses a simple templating system for substituting factors and other values into scripts. Factors 
that should be substituted is wrapped in `{% ... %}`.

Factors are substituted using their names specified in the design. 
An example, if the value of `FactorA` should be passed as argument to `my_script.sh` the script
specified in the job is written as `my_script.sh {% FactorA %}`. Special tags available for substitions
are:
* {% results_file %}: Will substitute the tag with the results-file for the current experiment.
* Other constants specified under `constants` not written in capital letters. 

Path-substitutions can be used to use files available during the current iteration
or other special directories. Values for path-substitution are indicated by capital letters
and are used as following `{% DIRNAME filename %}`. This will substitute the tag with `DIRNAME/filename`. 
Special paths available are:
* `BASEDIR`: Path to the root working-directory, i.e. `working_directory`.
* `WORKDIR`: Path to the current iterations working-directory.
* Other constants specified under `constants` written in capital letters.
