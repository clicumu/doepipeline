import re


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