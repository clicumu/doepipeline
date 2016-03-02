import yaml
import abc
import re


class BaseClient:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect(self, *args):
        pass

    @abc.abstractmethod
    def disconnect(self, *args):
        pass

    @abc.abstractmethod
    def execute_command(self, **args):
        pass


class PipelineRunner:

    def __init__(self, config, client):
        try:
            self._validate_config(config)
        except AssertionError:
            raise ValueError('Invalid config')

        if not isinstance(client, BaseClient):
            msg = 'client must be derived from BaseClient, not: {}'
            raise ValueError(msg.format(type(client)))

        self._client = client
        self._config = config
        self._current_iteration = 0

        try:
            env_variables = config['before_run'].pop('environment_variables')
        except KeyError:
            env_variables = None

        self._env_variables = env_variables

        jobs = [config[job] for job in config['pipeline']]
        self._scripts_templates = [parse_job_to_template_string(job)
                                   for job in jobs]
        self._factors = {key: factor['factor_name'] for job in jobs
                         for key, factor in job['factors'].items()}


    @classmethod
    def from_yaml(cls, path, *args, **kwargs):
        config = yaml.load(open(path))
        return cls(config, *args, **kwargs)

    def new_pipeline_collection(self, experiment_design):
        """ Given experiment, create script-strings to execute.

        Parameter settings from experimental design are used to
        render template script strings. Results are are returned
        in a dict with experiment number as key and list containing
        pipeline strings.

        Example output:
        pipeline_collection = {
            1: ['./script_one --param 1', './script_two --other-param 3'],
            2: ['./script_one --param 2', './script_two --other-param 4'],
            ...
        }

        :param experiment_design: Experimental design.
        :type experiment_design: pandas.DataFrame
        :return: Dictionary containing rendered script strings.
        :rtype: dict
        """
        pipeline_collection = dict()

        for _, experiment in experiment_design.iterrows():
            exp_no = experiment['Exp No']
            rendered_scripts = list()

            for script in self._scripts_templates:
                for factor, factor_name in self._factors.items():
                    # Get current parameter setting.
                    factor_value = experiment[factor_name]
                    replacement = {factor: factor_value}
                    try:
                        script = script.format(**replacement)
                    except KeyError:
                        # Current factor not present in current script.
                        continue

                rendered_scripts.append(script)

            pipeline_collection[exp_no] = rendered_scripts

        return pipeline_collection

    def _validate_config(self, config_dict):
        """ Input validation of config.

        Raises AssertionError if config is invalid.

        :param config_dict: Pipeline configuration.
        :raises: AssertionError
        """
        reserved_terms = 'before_run', 'pipeline'
        valid_before = 'environment_variables'
        assert 'pipeline' in config_dict

        job_names = config_dict['pipeline']
        assert isinstance(job_names, list)
        assert all(job in config_dict for job in job_names)
        assert all(term in job_names for term in config_dict
                   if term not in reserved_terms)

        jobs = [config_dict[job_name] for job_name in job_names]

        # Check existence of scripts and that they are simple strings.
        assert all('script' in job for job in jobs)
        assert all(isinstance(job['script'], str) for job in jobs)

        job_w_factors = [job for job in jobs if 'factors' in job]

        # Check factors are dicts.
        assert all(all(isinstance(factor, dict) for factor in job['factors'].values())\
                   for job in job_w_factors)
        # Check factors either script_option or substituted.
        assert all(any(['script_option' in factor, 'substitute' in factor])\
                       for factor in job['factors'].values() for job in job_w_factors)

        # Get jobs with substitution
        job_w_sub = [job for job in job_w_factors if\
                     any('substitute' in factor for factor in job['factors'].values())]

        # Assert that substitution-factors can be substituted.
        for job in job_w_sub:
            assert all(re.search(r'{%\s*' + fac + r'\s*%}', job['script'])\
                       for fac, fac_d in job['factors'].items() if fac_d.get('substitute', False))

        assert all(key in valid_before for key in config_dict['before_run'])


def parse_job_to_template_string(job):
    """ Parse config job-entry into template string.

    :param job: Config entry for job.
    :type job: dict
    :return: Parsed string
    :rtype: str
    """
    script = job['script'].strip()

    try:
        factors = job['factors']
    except KeyError:
        # Script with no additional factors
        pass
    else:
        for key, factor in factors.items():
            if factor.get('script_option', False):
                option_ = factor['script_option']
                script += ' %s {%s}' % (option_, key)
            if factor.get('substitute', False):
                template_pattern = r'{%\s*' + key + r'\s*%}'
                script = re.sub(template_pattern, '{' + key + '}', script)

    return script