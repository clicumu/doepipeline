import yaml
import re
import collections
import os

from doepipeline.designer import BaseExperimentDesigner, ExperimentDesigner
from doepipeline.utils import parse_job_to_template_string


class PipelineGenerator:

    """
    Generator class for pipelines.

    Given config the :class:`PipelineGenerator` produces template
    scripts. When given an experimental design the scripts are
    rendered into ready-to-run script strings.
    """

    def __init__(self, config, path_sep=None):
        try:
            self._validate_config(config)
        except AssertionError as e:
            raise ValueError('Invalid config: ' + str(e))

        self._config = config
        self._current_iteration = 0

        before = config.get('before_run', {})
        self._env_variables = before.get('environment_variables', None)
        self._setup_scripts = before.get('scripts', None)

        jobs = [config[job] for job in config['pipeline']]
        specials = {'results_file': config['results_file'],
                    'WORKDIR': config.get('working_directory', '.')}
        self._scripts_templates = [
            parse_job_to_template_string(job, specials, path_sep) for job in jobs
        ]
        self._factors = config['design']['factors']

    @classmethod
    def from_yaml(cls, yaml_config, *args, **kwargs):
        if isinstance(yaml_config, str):
            with open(yaml_config) as f:
                try:
                    config = yaml.load(f)
                except yaml.parser.ParserError:
                    raise ValueError('config not valid YAML')
        else:
            try:
                config = yaml.load(yaml_config)
            except AttributeError:
                raise ValueError('yaml_config must be path or file-handle')

        return cls(config, *args, **kwargs)

    def new_designer_from_config(self, designer_class=None, *args, **kwargs):
        if designer_class is None:
            designer_class = ExperimentDesigner

        factors = self._config['design']['factors']
        design_type = self._config['design']['type']
        responses = self._config['design']['responses']

        return designer_class(factors, design_type, responses, *args, **kwargs)

    def new_pipeline_collection(self, experiment_design, exp_id_column=None):
        """ Given experiment, create script-strings to execute.

        Parameter settings from experimental design are used to
        render template script strings. Results are are returned
        in an :class:`OrderedDict` with experiment indexes as key
        and list containing pipeline strings.

        Example output:
        pipeline_collection = {
            '0': ['./script_one --param 1', './script_two --other-param 3'],
            '1': ['./script_one --param 2', './script_two --other-param 4'],
            ...
        }

        :param experiment_design: Experimental design.
        :type experiment_design: pandas.DataFrame
        :param exp_id_column: Column of experimental identifiers.
        :type exp_id_column: str | None
        :return: Dictionary containing rendered script strings.
        :rtype: collections.OrderedDict
        """
        pipeline_collection = collections.OrderedDict()

        for i, experiment in experiment_design.iterrows():
            if exp_id_column is not None:
                exp_id = str(experiment[exp_id_column])
            else:
                exp_id = str(i)

            rendered_scripts = list()

            for script in self._scripts_templates:
                # Find which factors are used in the script template
                factor_name_list = [factor_name for factor_name in self._factors]
                pattern = re.compile("(" + "|".join(factor_name_list) + ")")
                script_factors = re.findall(pattern, script)

                # Get current factor settings
                replacement = {}
                for factor_name in script_factors:
                    factor_value = experiment[factor_name]
                    replacement[factor_name] = int(round(factor_value))

                # Replace the factor placeholders with the factor values
                script = script.format(**replacement)

                rendered_scripts.append(script)

            pipeline_collection[exp_id] = rendered_scripts

        pipeline_collection['ENV_VARIABLES'] = self._env_variables
        pipeline_collection['SETUP_SCRIPTS'] = self._setup_scripts
        pipeline_collection['RESULTS_FILE'] = self._config['results_file']
        pipeline_collection['WORKDIR'] = self._config.get('working_directory', '.')
        pipeline_collection['JOBNAMES'] = self._config['pipeline']

        if 'SLURM' in self._config:
            slurm = self._config['SLURM']
            slurm['jobs'] = list()
            for job in (self._config[name] for name in self._config['pipeline']):
                slurm['jobs'].append(job.get('SLURM', None))

            pipeline_collection['SLURM'] = slurm

        return pipeline_collection

    def _validate_config(self, config_dict):
        """ Input validation of config.

        Raises AssertionError if config is invalid.

        :param config_dict: Pipeline configuration.
        :raises: AssertionError
        """
        reserved_terms = ('before_run', 'pipeline', 'design',
                          'results_file', 'working_directory', 'SLURM')
        valid_before = 'environment_variables', 'scripts'
        assert 'pipeline' in config_dict, 'pipeline missing'
        assert 'design' in config_dict, 'design missing'
        assert 'results_file', 'collect_results missing'

        job_names = config_dict['pipeline']
        assert isinstance(job_names, list), 'pipeline must be listing'
        assert all(job in config_dict for job in job_names),\
            'all jobs in pipeline must be specified'
        assert all(term in job_names for term in config_dict
                   if term not in reserved_terms), 'all specified jobs must be in pipeline'

        if 'before_run' in config_dict:
            before = config_dict['before_run']
            assert all(key in valid_before for key in before),\
                'invalid key, allowed before_run: {}'.format(', '.join(valid_before))
            if 'scripts' in before:
                assert isinstance(before['scripts'], list),\
                    'before_run scripts must be a list of strings'
                assert all(isinstance(script, str)\
                           for script in before['scripts']),\
                    'before_run scripts must be a list of strings'
            if 'environment_variables' in before:
                assert isinstance(before['environment_variables'], dict),\
                    'environment_variables must be key-value-pairs'
                assert all(isinstance(value, str) for value\
                           in before['environment_variables'].values()),\
                    'environment_variables values must be strings'

        design = config_dict['design']
        allowed_factor_keys = 'min', 'max', 'low_init', 'high_init', 'type'
        assert 'type' in design, 'design type is missing'
        assert 'factors' in design, 'design factors is missing'
        assert 'responses' in design, 'design responses is missing'
        design_factors = design['factors']
        design_responses = design['responses']
        # Check that factors are specified.
        for key, factor_settings in design_factors.items():
            assert all(key in allowed_factor_keys for key in factor_settings),\
                'invalid key, allowed keys for factors: {}'.format(allowed_factor_keys)

        # Check that responses are specified.
        assert isinstance(design_responses, dict),\
            'design responses must be key-value-pairs'
        assert all(isinstance(target, dict) for target in design_responses.values()),\
            'design responses optimization goal must be key-value mappings'

        jobs = [config_dict[job_name] for job_name in job_names]

        # Check existence of scripts and that they are simple strings.
        assert all('script' in job for job in jobs), 'all jobs must have script'
        assert all(isinstance(job['script'], str) for job in jobs),\
            'job scripts must be strings'

        job_w_factors = [job for job in jobs if 'factors' in job]

        # Check factors are dicts.
        assert all(all(isinstance(factor, dict) for factor in job['factors'].values())\
                   for job in job_w_factors), 'job factors must be key-value-pairs'
        assert all(all(factor in design_factors for factor in job['factors'].keys())\
                   for job in job_w_factors), 'job factors must be specified in design'

        # Check factors either script_option or substituted.
        for job in job_w_factors:
            assert any(['script_option' in factor, 'substitute' in factor]\
                       for factor in job['factors']),\
            'factors must be added as script option or substituted'

        # Get jobs with substitution
        job_w_sub = [job for job in job_w_factors if\
                     any('substitute' in factor for factor in job['factors'].values())]

        # Assert that substitution-factors can be substituted.
        for job in job_w_sub:
            msg = 'substituted factors must be templated in script-string'
            assert all(re.search(r'{%\s*' + fac + r'\s*%}', job['script'])\
                       for fac, fac_d in job['factors'].items()\
                       if fac_d.get('substitute', False)), msg

        # Check SLURM-specifics.
        if any('SLURM' in job.keys() for job in jobs):
            assert 'SLURM' in config_dict, \
                "at least one job specified with SLURM but no SLURM entry found in config file"

        if 'SLURM' in config_dict:
            assert 'account_name' in config_dict['SLURM'],\
                'SLURM account name required'

            for job in (j for j in jobs if 'SLURM' in j):
                assert 'SLURM' in job,\
                    'All jobs must have SLURM-settings specified'
                assert 'p' in job['SLURM'] and job['SLURM']['p'],\
                    'job type must be set (-p)'
                assert 'n' in job['SLURM'] and job['SLURM']['n'],\
                    'job cores/nodes must be set (-n)'
                assert 't' in job['SLURM'] and job['SLURM']['t'],\
                    'job time must be set (-t)'
