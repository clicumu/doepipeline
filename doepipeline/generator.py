import re
import collections
import os
import logging

import yaml
import numpy as np

from doepipeline.designer import ExperimentDesigner
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
        self._current_iteration = 1
        self._setting_up = True

        before = config.get('before_run', {})
        self._env_variables = before.get('environment_variables', None)
        self._setup_scripts = before.get('scripts', None)

        jobs = [config[job] for job in config['pipeline']]

        # current workdir should be corresponding iteration, but save the base directory
        self._config['base_directory'] = self._config['working_directory']
        self._update_working_directory()

        specials = {'results_file': self._config['results_file'],
                    'WORKDIR': self._config['working_directory'],
                    'BASEDIR': self._config['base_directory']}
        specials.update(self._config.get('constants', dict()))

        self._scripts_templates = [
            parse_job_to_template_string(job, specials, path_sep) for job in jobs
        ]
        self._factors = config['design']['factors']

    def _update_working_directory(self):
        workdir = os.path.join(self._config.get('base_directory', '.'), str(self._current_iteration))
        logging.debug('New working directory: {}'.format(workdir))
        self._config['working_directory'] = workdir

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

    def get_base_directory(self):
        return self._config['base_directory']

    def set_current_iteration(self, iter):
        self._current_iteration = iter
        self._update_working_directory()

    def new_pipeline_collection(self, experiment_design,
                                exp_id_column=None, validation_run=False):
        """ Given experiment, create script-strings to execute.

        Parameter settings from experimental design are used to
        render template script strings. Results are are returned
        in an :class:`OrderedDict` with experiment indexes as key
        and list containing pipeline strings.

        Example output:
        pipeline_collection = {
            0: ['./script_one --param 1', './script_two --other-param 3'],
            1: ['./script_one --param 2', './script_two --other-param 4'],
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
        if not self._setting_up and not validation_run:
            self._current_iteration += 1
            logging.debug('generator.py: incrementing _current_iteration. '
                          'Is now {}'.format(self._current_iteration))
            self._update_working_directory()

        for i, experiment in experiment_design.iterrows():
            if exp_id_column is not None:
                exp_id = experiment[exp_id_column]
            else:
                exp_id = i

            rendered_scripts = list()

            for script in self._scripts_templates:

                # Find which factors are used in the script template
                factor_name_list = [factor_name for factor_name in self._factors]
                pattern = re.compile("(" + "|".join(factor_name_list) + ")")
                script_factors = re.findall(pattern, script)

                # Get current factor settings
                replacement = {}
                for factor_name in script_factors:
                    factor_type = self._factors[factor_name].get('type', 'quantitative')
                    factor_value = experiment[factor_name]
                    replacement[factor_name] = int(factor_value) if \
                        factor_type.lower() == 'ordinal' else factor_value

                # Replace the factor placeholders with the factor values
                script = script.format(**replacement)

                rendered_scripts.append(script)

            pipeline_collection[exp_id] = rendered_scripts

        pipeline_collection['ENV_VARIABLES'] = self._env_variables
        pipeline_collection['SETUP_SCRIPTS'] = self._setup_scripts
        pipeline_collection['RESULTS_FILE'] = self._config['results_file']
        pipeline_collection['WORKDIR'] = self._config['working_directory']
        pipeline_collection['JOBNAMES'] = self._config['pipeline']

        if self._setting_up:
            logging.debug('generator.py: _setting_up = False')
            self._setting_up = False

        jobs = [self._config[name] for name in self._config['pipeline']]

        if any('SLURM' in job for job in jobs):
            slurm = {'jobs': list()}

            for job in jobs:
                slurm['jobs'].append(job.get('SLURM', None))

            pipeline_collection['SLURM'] = slurm

        return pipeline_collection

    def _validate_config(self, config_dict):
        """ Input validation of config.

        Raises AssertionError if config is invalid.

        :param config_dict: Pipeline configuration.
        :raises: AssertionError
        """
        reserved_terms = ('before_run', 'pipeline', 'design', 'constants',
                          'results_file', 'working_directory')
        valid_before = 'environment_variables', 'scripts'

        assert 'pipeline' in config_dict, 'pipeline missing'
        assert 'design' in config_dict, 'design missing'
        assert 'results_file', 'results_file missing'
        assert 'working_directory' in config_dict, 'working directory missing'

        _validate_constants(config_dict)

        job_names = config_dict['pipeline']
        jobs = [config_dict[job_name] for job_name in job_names]

        _validate_job_list_config(config_dict, job_names, reserved_terms)
        _validate_setup_scrip_config(config_dict, valid_before)

        design = config_dict['design']

        assert 'type' in design, 'design type is missing'
        assert 'factors' in design, 'design factors is missing'
        assert 'responses' in design, 'design responses is missing'

        if 'screening_reduction' in design:
            reduction = design['screening_reduction']
            assert reduction == 'auto' or \
                   (isinstance(reduction, int) and reduction > 1), \
                'screening_reduction must be "auto" or integer larger than 1.'

        design_factors = design['factors']
        design_responses = design['responses']

        _validate_factor_config(jobs, design_factors)
        _validate_response_config(design_responses)

        # SLURM-specifics.
        for job in (j for j in jobs if 'SLURM' in j):
            assert isinstance(job['SLURM'], dict), \
                'SLURM-config must be mapping.'


def _validate_constants(config_dict):
    constants = config_dict.get('constants', dict())
    assert isinstance(constants, dict), 'constants must be key-value mapping'
    assert all(key.isupper() for key in constants.keys()), \
        'constant-keys must be upper-case'


def _validate_job_list_config(config_dict, job_names, reserved_terms):
    assert isinstance(job_names, list), 'pipeline must be listing'
    assert all(job in config_dict for job in job_names), \
        'all jobs in pipeline must be specified'
    assert all(term in job_names for term in config_dict
               if
               term not in reserved_terms), 'all specified jobs must be in pipeline'


def _validate_response_config(design_responses):
    # Check that responses are specified.
    assert isinstance(design_responses, dict), \
        'design responses must be key-value-pairs'
    assert all(
        isinstance(target, dict) for target in design_responses.values()), \
        'design responses optimization goal must be key-value mappings'
    has_multiple = len(design_responses) > 1
    for name, response_spec in design_responses.items():
        assert 'criterion' in response_spec, 'response {} have not specified ' \
                                             '"criterion".'.format(name)
        criterion = response_spec['criterion']
        assert criterion in ('maximize', 'minimize', 'target'), \
            'response {} has invalid criterion {} (valid options "maximize", ' \
            '"minimize", "target")'.format(name, criterion)

        if has_multiple:
            if criterion == 'target':
                required_terms = ('low_limit', 'high_limit', 'target')
            elif criterion == 'maximize':
                required_terms = ('low_limit', 'target')
            else:
                required_terms = ('high_limit', 'target')

            for required in required_terms:
                assert required in response_spec, \
                    'response {} is missing "{}" which is required for responses ' \
                    'with criterion "{}" when multiple responses are used'.format(
                        name, required, criterion)
                assert isinstance(response_spec[required], (np.number, int, float)), \
                    '{} for response {} is not numeric.'.format(name, required)

        if 'transform' in response_spec:
            transform = response_spec['transform']
            assert transform in ('log', 'box-cox'), \
                'response {} has invalid transform {} (valid options "log", ' \
                '"box-cox").'.format(name, transform)


def _validate_factor_config(jobs, design_factors):
    allowed_factor_keys = 'min', 'max', 'low_init', 'high_init', \
                          'type', 'values', 'screening_levels'

    allowed_factor_types = 'quantitative', 'ordinal', 'categorical'

    # Check that factors are specified.
    for key, factor_settings in design_factors.items():
        assert all(key in allowed_factor_keys for key in factor_settings), \
            'invalid key in {}, allowed keys for factors: {}'.format(
                key, allowed_factor_keys)
        if 'type' in factor_settings:
            assert factor_settings['type'].lower() in allowed_factor_types, \
                '"type" must be one of {}, error in factor {}'.format(
                    allowed_factor_types, key)

        if 'screening_levels' in factor_settings:
            factor_type = factor_settings.get('type', 'continuous').lower()
            assert factor_type != 'categorical', \
                'screening_levels can\'t be set for categorical factors'
            levels = factor_settings['screening_levels']
            assert isinstance(levels, int) and levels > 1, \
                'screening_levels must be integer larger than 1.'

    # Check existence of scripts and that they are simple strings.
    assert all('script' in job for job in jobs), 'all jobs must have script'
    assert all(isinstance(job['script'], str) for job in jobs), \
        'job scripts must be strings'
    job_w_factors = [job for job in jobs if 'factors' in job]

    # Check factors are dicts.
    assert all(
        all(isinstance(factor, dict) for factor in job['factors'].values()) \
        for job in job_w_factors), 'job factors must be key-value-pairs'
    assert all(
        all(factor in design_factors for factor in job['factors'].keys()) \
        for job in job_w_factors), 'job factors must be specified in design'
    # Check factors either script_option or substituted.
    for job in job_w_factors:
        assert any(['script_option' in factor, 'substitute' in factor] \
                   for factor in job['factors']), \
            'factors must be added as script option or substituted'

    # Get jobs with substitution
    job_w_sub = [job for job in job_w_factors if \
                 any('substitute' in factor for factor in
                     job['factors'].values())]
    # Assert that substitution-factors can be substituted.
    for job in job_w_sub:
        msg = 'substituted factors must be templated in script-string'
        assert all(re.search(r'{%\s*' + fac + r'\s*%}', job['script']) \
                   for fac, fac_d in job['factors'].items() \
                   if fac_d.get('substitute', False)), msg


def _validate_setup_scrip_config(config_dict, valid_before):
    if 'before_run' in config_dict:
        before = config_dict['before_run']
        assert all(key in valid_before for key in before), \
            'invalid key, allowed before_run: {}'.format(
                ', '.join(valid_before))
        if 'scripts' in before:
            assert isinstance(before['scripts'], list), \
                'before_run scripts must be a list of strings'
            assert all(isinstance(script, str) \
                       for script in before['scripts']), \
                'before_run scripts must be a list of strings'
        if 'environment_variables' in before:
            assert isinstance(before['environment_variables'], dict), \
                'environment_variables must be key-value-pairs'
            assert all(isinstance(value, str) for value \
                       in before['environment_variables'].values()), \
                'environment_variables values must be strings'
