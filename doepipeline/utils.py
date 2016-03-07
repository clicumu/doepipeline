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


def validate_command(command):
    """ Validate command-string.

    Example input:
    >>> command = 'nohup {script}'
    >>> command = '{script} > {logfile}'

    Required tags:

    * script
    
    Allowed tags:

    * script
    * logfile

    :param command: Command string-template.
    :return: Stripped input template if validation succeeds.
    :raises: AssertionError
    """
    assert isinstance(command, str), 'base command must be string'
    command = command.strip()

    try:
        _validate_string(command, ['script'], ['script', 'logfile'])
    except AssertionError:
        raise
    else:
        return command


def validate_log_file(log_file_name):
    """ Validate base log-file name.

    Example input:
    >>> logfile = '{name}_{i}.log'

    Required (and allowed) tags:

    * name
    * i

    :param log_file_name: Input log-file name.
    :return: Stripped input template if validation succeeds.
    :raises: AssertionError
    """
    assert isinstance(log_file_name, str), 'base-log-file must be string'
    log_file_name = log_file_name.strip()
    try:
        _validate_string(log_file_name, ['name', 'i'], ['name', 'i'])
    except AssertionError:
        raise
    else:
        return log_file_name


def _validate_string(input_string, required_tags, allowed_tags):
    """ Check input for required and allowed string template tags.

    :param str input_string: String to validate.
    :param list required_tags: Sequence of required tags.
    :param list allowed_tags: Sequence of allowed tags.
    :raises: AssertionError
    """
    tags = re.findall(r'{([a-zA-Z0-9_\s]*)}', input_string)

    for required_tag in required_tags:
        assert required_tag in tags, '"{}"-tag missing'.format(required_tag)

    for tag in tags:
        assert tag in allowed_tags, 'unallowed tag: ' + tag