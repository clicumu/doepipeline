import re
import os


def parse_job_to_template_string(job, specials=None, path_sep=None):
    """ Parse config job-entry into template string.

    :param dict job: Config entry for job.
    :param dict specials: Non-job related substitution entries.
    :param str path_sep: Path-separator to use. Defaults to local separator.
    :return: Parsed string
    :rtype: str
    """
    if path_sep is None:
        path_sep = os.path.sep

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

    specials = specials if specials is not None else {}
    for key, value in specials.items():
        if key.isupper():
            script = substitute_path(script, key, value, path_sep)
        else:
            template_pattern = r'{%\s*' + key + r'\s*%}'
            script = re.sub(template_pattern, value, script)
    return script


def substitute_path(template_str, key, path, path_sep):
    """

    :param str template_str:
    :param str key:
    :param str path:
    :param str path_sep:
    :return: Formatted string.
    """
    template_pattern = r'{%\s*' + key + r'\s*(\S*)\s*%}'
    match = re.search(template_pattern, template_str)

    if match is not None:
        ending = match.groups()[0]
        full_path = path_sep.join([path, ending])
        result = template_str.replace(match.group(), full_path)
    else:
        result = template_str

    return result


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