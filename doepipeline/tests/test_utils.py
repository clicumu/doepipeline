import unittest
from doepipeline.utils import parse_job_to_template_string


class TestJobParse(unittest.TestCase):

    def make_job(self, script, factors=None):
        """ Make job dictionary given script and factors.

        :param script: Script-string.
        :param factors: Factor-dict.
        :return: Config job-dict.
        """
        job = {'script': script}
        if factors is not None:
            job['factors'] = factors
        return job

    def make_factor(self, name, substitute=False, option=None):
        """ Given name, and type of factor make factor dict.

        :param name: factor_name
        :param substitute: True if template factor.
        :param option: Option name
        :return: factor-dict
        """
        if not (substitute or option):
            raise ValueError

        factor = {'factor_name': name}
        if substitute:
            factor['substitute'] = True
        else:
            factor['script_option'] = option

        return factor

    def test_parse_job_with_no_factors_return_script(self):
        script = './really_cool_script --with options'
        job = self.make_job(script)
        parsed_job = parse_job_to_template_string(job)
        self.assertEqual(parsed_job, script)

    def test_parse_job_with_substitution(self):
        script = './script {% Factor %}'
        factors = {'Factor': self.make_factor('name', True)}
        job = self.make_job(script, factors)
        parsed_job = parse_job_to_template_string(job)

        self.assertEqual(parsed_job, './script {Factor}')

    def test_parse_job_with_option(self):
        script = './script'
        factors = {'Factor': self.make_factor('name', False, '--opt')}
        job = self.make_job(script, factors)
        parsed_job = parse_job_to_template_string(job)

        self.assertEqual(parsed_job, './script --opt {Factor}')
