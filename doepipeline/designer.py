import logging
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import scipy.stats
import pyDOE2

from doepipeline.model_utils import make_desirability_function, predict_optimum


class OptimizationResult(namedtuple(
    'OptimizationResult', ['predicted_optimum', 'converged',
                           'tol', 'reached_limits', 'empirically_found'])):
    """ `namedtuple` encapsulating results from optimization. """


class UnsupportedFactorType(Exception):
    pass


class UnsupportedDesign(Exception):
    pass


class DesignerError(Exception):
    pass


class NumericFactor:

    """ Base class for numeric factors.

    Simple class which encapsulates current settings and allowed
    max and min.

    Can't be instantiated.
    """

    def __init__(self, factor_max, factor_min, current_low=None, current_high=None):
        if type(self) == NumericFactor:
            raise TypeError('NumericFactor can not be instantiated. Use '
                            'sub-classes instead.')
        self.current_low = current_low
        self.current_high = current_high
        self.max = factor_max
        self.min = factor_min
        self.screening_levels = 5

    @property
    def span(self):
        """ Distance between current high and low. """
        return self.current_high - self.current_low

    @property
    def center(self):
        """ Mean value of current high and low. """
        return (self.current_high + self.current_low) / 2.0

    def __repr__(self):
        return ('{}(factor_max={}, factor_min={}, current_low={}, '
                'current_high={})').format(self.__class__.__name__,
                                           self.max,
                                           self.min,
                                           self.current_low,
                                           self.current_high)


class QuantitativeFactor(NumericFactor):

    """ Real value factors. """


class OrdinalFactor(NumericFactor):

    """ Ordinal (integer) factors.

    Attributes are checked to be integers (or None/inf if allowed).
    """

    def __setattr__(self, attribute, value):
        """ Check values `current_low`, `current_high`, `max` and `min`.

        :param str attribute: Attribute name
        :param Any value: New value
        """
        numeric_attributes = ('current_low', 'current_high',
                              'max', 'min')
        if attribute in numeric_attributes:
            err_msg = '{} requires an integer, not {}'.format(attribute, value)
            if attribute == 'max' and value == float('inf'):
                pass
            elif attribute == 'min' and value == float('-inf'):
                pass
            elif isinstance(value, float) and not value.is_integer():
                raise ValueError(err_msg)
            elif isinstance(value, (float, int)):
                value = int(value)
            elif attribute in ('current_low', 'current_high') and value is None:
                pass
            else:
                raise ValueError(err_msg)

        super(OrdinalFactor, self).__setattr__(attribute, value)


class CategoricalFactor:

    """ Multilevel categorical factors. """

    def __init__(self, values, fixed_value=None):
        self.values = values
        self.fixed_value = fixed_value

    def __repr__(self):
        return '{}(values={}, fixed_value={})'.format(self.__class__.__name__,
                                                      self.values,
                                                      self.fixed_value)


class ExperimentDesigner:

    _matrix_designers = {
        'fullfactorial2levels': pyDOE2.ff2n,
        'fullfactorial3levels': lambda n: pyDOE2.fullfact([3] * n),
        'placketburman': pyDOE2.pbdesign,
        'boxbehnken': lambda n: pyDOE2.bbdesign(n, 1),
        'ccc': lambda n: pyDOE2.ccdesign(n, (0, 3), face='ccc'),
        'ccf': lambda n: pyDOE2.ccdesign(n, (0, 3), face='ccf'),
        'cci': lambda n: pyDOE2.ccdesign(n, (0, 3), face='cci'),
    }

    def __init__(self, factors, design_type, responses, skip_screening=True,
                 at_edges='distort', relative_step=.25, gsd_reduction='auto',
                 model_selection='brute', n_folds='loo', manual_formula=None,
                 shrinkage=1.0, q2_limit=0.5, gsd_span_ratio=0.5):
        try:
            assert at_edges in ('distort', 'shrink'),\
                'unknown action at_edges: {0}'.format(at_edges)
            assert relative_step is None or 0 < relative_step < 1,\
                'relative_step must be float between 0 and 1 not {}'.format(relative_step)
            assert model_selection in ('brute', 'greedy', 'manual'), \
                'model_selection must be "brute", "greedy", "manual".'
            assert n_folds == 'loo' or (isinstance(n_folds, int) and n_folds > 0), \
                'n_folds must be "loo" or positive integer'
            assert 0.9 <= shrinkage <= 1, 'shrinkage must be float between 0.9 and 1.0, not {}'.format(shrinkage)
            assert 0 <= q2_limit <= 1, 'q2_limit must be float between 0 and 1, not {}'.format(q2_limit)
            if model_selection == 'manual':
                assert isinstance(manual_formula, str), \
                    'If model_selection is "manual" formula must be provided.'
        except AssertionError as e:
            raise ValueError(str(e))

        self.factors = OrderedDict()
        factor_types = list()
        for factor_name, f_spec in factors.items():
            factor = factor_from_spec(f_spec)
            if isinstance(factor, CategoricalFactor) and skip_screening:
                raise DesignerError('Can\'t perform optimization with categorical '
                                    'variables without prior screening.')

            self.factors[factor_name] = factor
            logging.debug('Sets factor {}: {}'.format(factor_name, factor))

            factor_types.append(f_spec.get('type', 'continuous'))

        self.skip_screening = skip_screening
        self.step_length = relative_step
        self.design_type = design_type
        self.responses = responses
        self.response_values = None
        self.gsd_reduction = gsd_reduction
        self.model_selection = model_selection
        self.n_folds = n_folds
        self.shrinkage = shrinkage
        self.q2_limit = q2_limit
        self._formula = manual_formula
        self._edge_action = at_edges
        self._allowed_phases = ['optimization', 'screening']
        self._phase = 'optimization' if self.skip_screening else 'screening'
        self._n_screening_evaluations = 0
        self._factor_types = factor_types
        self._gsd_span_ratio = gsd_span_ratio
        self._stored_transform = lambda x: x
        self._best_experiment = {
                'optimal_x': pd.Series([]),
                'optimal_y': None,
                'weighted_y': None}
        n = len(self.factors)
        try:
            self._matrix_designers[self.design_type.lower()]
        except KeyError:
            raise UnsupportedDesign(self.design_type)

        if len(self.responses) > 1:
            self._desirabilites = {name: make_desirability_function(factor)
                                   for name, factor in self.responses.items()}
        else:
            self._desirabilites = None

    def new_design(self):
        """

        :return: Experimental design-sheet.
        :rtype: pandas.DataFrame
        """
        if self._phase == 'screening':
            return self._new_screening_design(reduction=self.gsd_reduction)
        else:
            return self._new_optimization_design()

    def write_factor_csv(self, out_file):
        factors = list()
        idx = pd.Index(['fixed_value', 'current_low', 'current_high'])

        for name, factor in self.factors.items():
            current_min = None
            current_high = None
            fixed_value = None

            if issubclass(type(factor), NumericFactor):
                current_min = factor.current_low
                current_high = factor.current_high
            elif isinstance(factor, CategoricalFactor):
                fixed_value = factor.fixed_value
            else:
                raise NotImplementedError

            data = [fixed_value, current_min, current_high]
            factors.append(pd.Series(data, index=idx, name=name))

        factors_df = pd.DataFrame(factors)
        logging.info('Saving factor settings to {}'.format(out_file))
        factors_df.to_csv(out_file)

    def update_factors_from_csv(self, csv_file):
        factors_df = pd.DataFrame.from_csv(csv_file)
        logging.info('Reading factor settings from {}'.format(csv_file))

        for name, factor in self.factors.items():
            logging.info('Updating factor {}'.format(name))

            if issubclass(type(factor), NumericFactor):
                current_low = factors_df.loc[name]['current_low']
                current_high = factors_df.loc[name]['current_high']
                logging.info('Factor: {}. Setting current_low to {}'.format(name, current_low))
                logging.info('Factor: {}. Setting current_high to {}'.format(name, current_high))
                factor.current_low = current_low
                factor.current_high = current_high
            elif isinstance(factor, CategoricalFactor):
                if pd.isnull(factors_df.loc[name]['fixed_value']):
                    fixed_value = None
                    logging.info('Factor: {}. Had no fixed_value.'.format(name))
                else:
                    fixed_value = factors_df.loc[name]['fixed_value']
                    logging.info('Factor: {}. Setting fixed_value to {}.'.format(name, fixed_value))

                factor.fixed_value = fixed_value

    def get_optimal_settings(self, response):
        """
        Calculate optimal factor settings given response. Returns calculated
        optimum.

        If the current phase is 'screening': returns the factor settings of
        the best run and updates the current factor settings.

        If the current phase is 'optimization': returns the factor settings of
        the predicted optimum, but doesn't update current factor settings in
        case a validation step is to be run first

        :param pandas.DataFrame response: Response sheet.

        :returns: Calculated optimum.
        :rtype: OptimizationResult
        """

        self._response_values = response.copy()
        response = response.copy()

        # Perform any transformations or weigh together multiple responses:
        treated_response, criterion = self.treat_response(response)

        if self._phase == 'screening':
            # Find the best screening result and update factors accordingly
            self._screening_response = treated_response
            self._screening_criterion = criterion
            return self._evaluate_screening(treated_response, criterion,
                                            self._gsd_span_ratio)
        else:
            # Predict optimal parameter settings, but don't update factors
            return self._predict_optimum_settings(treated_response, criterion)

    def _update_best_experiment(self, result):
        update = False
        if self._best_experiment['optimal_x'].empty:
            update = True
        elif result['criterion'] == 'maximize':
            if result['weighted_response'] > self._best_experiment['weighted_y']:
                update = True
        elif result['criterion'] == 'minimize':
            if result['weighted_response'] < self._best_experiment['weighted_y']:
                update = True
        if update:
            self._best_experiment['optimal_x'] = result['factor_settings']
            self._best_experiment['optimal_y'] = result['response']
            self._best_experiment['weighted_y'] = result['weighted_response']

        return update

    def get_best_experiment(self, experimental_sheet, response_sheet, use_index=1):
        """
        Accepts an experimental design and the corresponding response values.

        Finds the best experiment and updates self._best_experiment.

        Returns the best experiment, to be used in fnc update_factors_from_optimum
        """
        assert isinstance(experimental_sheet, pd.core.frame.DataFrame), \
            'The input experimental sheet must be a pandas DataFrame'
        assert isinstance(response_sheet, pd.core.frame.DataFrame), \
            'The input response sheet must be a pandas DataFrame'
        assert sorted(experimental_sheet.columns) == sorted(self.factors), \
            'The factors of the experimental sheet must match those in the \
            pipeline. You input:\n{}\nThey should be:\n{}'.format(
                list(experimental_sheet.columns),
                list(self.factors.keys()))
        assert sorted(response_sheet.columns) == sorted(self.responses), \
            'The responses of the response sheet must match those in the \
            pipeline. You input:\n{}\nThey should be:\n{}'.format(
                list(response_sheet.columns),
                list(self.responses.keys()))

        response = response_sheet.copy()
        treated_response, criterion = self.treat_response(
            response, perform_transform=False)

        treated_response = treated_response.iloc[:, 0]
        if criterion == 'maximize':
            optimum_i = treated_response.argsort().iloc[-use_index]
        elif criterion == 'minimize':
            optimum_i = treated_response.argsort().iloc[use_index - 1]
        else:
            raise NotImplementedError

        optimum_settings = experimental_sheet.iloc[optimum_i]

        results = OrderedDict()
        optimal_weighted_response = np.array(treated_response.iloc[optimum_i])
        optimal_response = response_sheet.iloc[optimum_i]
        results['factor_settings'] = optimum_settings
        results['weighted_response'] = optimal_weighted_response
        results['response'] = optimal_response
        results['criterion'] = criterion
        results['new_best'] = False
        results['old_best'] = self._best_experiment

        has_multiple_responses = response_sheet.shape[1] > 1
        logging.debug('The best response was found in experiment:\n{}'.format(optimum_settings.name))
        logging.debug('The response values were:\n{}'.format(response_sheet.iloc[optimum_i]))
        if has_multiple_responses:
            logging.debug('The weighed response was:\n{}'.format(treated_response.iloc[optimum_i]))
        logging.debug('Will return optimum settings:\n{}'.format(results['factor_settings']))
        logging.debug('And best response:\n{}'.format(results['response']))

        if self._update_best_experiment(results):
            results['new_best'] = True

        return results

    def update_factors_from_optimum(self, optimal_experiment, tol=0.25, recovery=False):
        """
        Updates the factor settings based on how far the current settings are
        from those supplied in optimal_experiment['factor_settings'].

        :param OrderedDict optimal_experiment: Output from get_best_experiment
        :param float tol: Accepted relative distance to design space edge.
        :returns: Calculated optimum.
        :rtype: OptimizationResult
        """
        are_numeric = np.array(self._factor_types) != 'categorical'
        numeric_names = np.array(list(self.factors.keys()))[are_numeric]
        numeric_factors = np.array(list(self.factors.values()))[are_numeric]

        optimal_x = optimal_experiment['factor_settings']
        optimal_y = optimal_experiment['weighted_response']
        criterion = optimal_experiment['criterion']

        # Get only numeric factors
        if recovery:
            optimal_x = optimal_x.iloc[optimal_x.index.isin(numeric_names)]

        centers = np.array([f.center for f in numeric_factors])
        spans = np.array([f.span for f in numeric_factors])

        ratios = (optimal_x - centers) / spans
        if not recovery:
            logging.debug(
                'The distance of the factor optimas from the factor centers, '
                'expressed as the ratio of the step length:\n{}'.format(ratios))

        if (abs(ratios) < tol).all():
            converged = True
            if not recovery:
                logging.info('Convergence reached.')
        else:
            converged = False
            if not recovery:
                logging.info('Convergence not reached. Moves design.')

            for ratio, name, factor in zip(ratios, numeric_names, numeric_factors):
                if abs(ratio) < tol:
                    if not recovery:
                        logging.debug(('Factor {} not updated - within tolerance '
                            'limits.').format(name))
                    continue

                if not recovery:
                    self._update_numeric_factor(factor, name, ratio)

        converged, reached_limits = self._check_convergence(
            centers,
            converged,
            criterion,
            optimal_y,
            numeric_factors,
            recovery=recovery)

        optimization_results = pd.Series(
            index=self._design_sheet.columns,
            dtype=object)

        for name, factor in self.factors.items():
            if isinstance(factor, CategoricalFactor):
                optimization_results[name] = factor.fixed_value
            else:
                optimization_results[name] = optimal_x[name]

        results = OptimizationResult(
            optimization_results,
            converged,
            tol,
            reached_limits,
            empirically_found=True)

        return results

    def _predict_optimum_settings(self, response, criterion):
        """
        Calculate a model from the response and find the optimum.

        :returns: Calculated optimum.
        :rtype: OptimizationResult
        """
        logging.info('Predicting optimum')

        are_numeric = np.array(self._factor_types) != 'categorical'
        numeric_names = np.array(list(self.factors.keys()))[are_numeric]

        optimal_x, model, prediction = predict_optimum(
            self._design_sheet.loc[:, are_numeric],
            response.iloc[:, 0].values,
            numeric_names,
            criterion=criterion,
            n_folds=self.n_folds,
            model_selection=self.model_selection,
            manual_formula=self._formula,
            q2_limit=self.q2_limit)

        optimization_results = pd.Series(
            index=self._design_sheet.columns,
            dtype=object)
        if not optimal_x.empty:
            # If Q2 of model was above the limit and if an optimum was found
            for name, factor in self.factors.items():
                if isinstance(factor, CategoricalFactor):
                    optimization_results[name] = factor.fixed_value
                elif isinstance(factor, OrdinalFactor):
                    optimization_results[name] = int(np.round(optimal_x[name]))
                else:
                    optimization_results[name] = optimal_x[name]

        result = OptimizationResult(
            optimization_results,
            converged=False,
            tol=0,
            reached_limits=False,
            empirically_found=False)

        return result

    def treat_response(self, response, perform_transform=True):
        """
        Perform any specified transformations on the response.

        If several responses are defined, combine them into one. The geometric
        mean of Derringer and Suich's desirability functions will be used for
        optimization, see:

        Derringer, G., and Suich, R., (1980), "Simultaneous Optimization
        of Several Response Variables," Journal of Quality Technology, 12,
        4, 214-219.

        Returns a single response variable and the associated maximize/minimize
        criterion.
        """

        has_multiple_responses = response.shape[1] > 1
        for name, spec in self.responses.items():
            transform = spec.get('transform', None)
            response_values = response[name]
            if perform_transform:
                if transform == 'log':
                    logging.debug('Log-transforming response {}'.format(name))
                    response_values = np.log(response_values)
                    self._stored_transform = np.log
                elif transform == 'box-cox':
                    response_values, lambda_ = scipy.stats.boxcox(response_values)
                    logging.debug('Box-cox transforming response {} '
                                  '(lambda={:.4f})'.format(name, lambda_))
                    self._stored_transform = _make_stored_boxcox(lambda_)
                else:
                    self._stored_transform = lambda x: x

            if has_multiple_responses:
                desirability_function = self._desirabilites[name]
                response_values = [desirability_function(value)
                                   for value in response_values]
            response[name] = response_values

        if has_multiple_responses:
            response = np.power(response.product(axis=1), (1 / response.shape[1]))
            response = response.to_frame('combined_response')
            criterion = 'maximize'

        else:
            criterion = list(self.responses.values())[0]['criterion']

        return response, criterion

    def reevaluate_screening(self):
        if self._screening_response is None:
            raise DesignerError('screening must be run before re-evaluation')

        return self._evaluate_screening(self._screening_response,
                                        self._screening_criterion,
                                        self._gsd_span_ratio,
                                        self._n_screening_evaluations + 1)

    def _validate_new_factor_limits(self, factor, factor_name, low_limit, high_limit):
        # If the proposed step change takes us below or above min and max:
        logging.debug('Factor {}: Proposed new factor low is {}.'
            .format(factor_name, low_limit))
        logging.debug('Factor {}: Proposed new factor high is {}.'
            .format(factor_name, high_limit))
        adjusted_settings = False
        if low_limit < factor.min:
            nudge = abs(low_limit - factor.min)
            logging.debug(
                'Factor {}: Minimum allowed setting ({}) would be exceeded by '
                'the proposed new factor low.'.format(factor_name, factor.min))
            low_limit += nudge
            high_limit += nudge
            adjusted_settings = True

        elif high_limit > factor.max:
            nudge = abs(high_limit - factor.max)
            logging.debug(
                'Factor {}: Maximum allowed setting ({}) would be exceeded by '
                'the proposed new factor high.'.format(factor_name, factor.max))
            low_limit -= nudge
            high_limit -= nudge
            adjusted_settings = True

        if adjusted_settings:
            logging.debug('Factor {}: Adjusted the proposed new factor '
                          'settings by {}.'.format(factor_name, nudge))
            logging.debug('Factor {}: New factor low is {}.'.format(factor_name, low_limit))
            logging.debug('Factor {}: New factor high is {}.'.format(factor_name, high_limit))

        return (low_limit, high_limit)

    def _evaluate_screening(self, response, criterion, span_ratio, use_index=1):
        """
        :param float span_ratio: The ratio of the span between gsd points that will be used in the following optimization design.
        """
        self._n_screening_evaluations += 1

        logging.info('Evaluating screening results.')
        response_series = response.iloc[:, 0]
        factor_items = sorted(self.factors.items())
        if criterion == 'maximize':
            optimum_i = response_series.argsort().iloc[-use_index]
        elif criterion == 'minimize':
            optimum_i = response_series.argsort().iloc[use_index - 1]
        else:
            raise NotImplementedError

        optimum_design_row = self._design_matrix[optimum_i]
        optimum_settings = OrderedDict()

        # Update all factors according to current results. For each factor,
        # the current_high and current_low will be set to factors level above
        # and below the point in the screening design with the best response.
        for factor_level, (name, factor) in zip(optimum_design_row, factor_items):
            if isinstance(factor, CategoricalFactor):
                factor_levels = np.array(factor.values)
                factor.fixed_value = factor_levels[factor_level]
            else:
                factor_levels = sorted(self._design_sheet[name].unique())

                min_ = factor_levels[max([0, factor_level - 1])]
                max_ = factor_levels[min([factor_level + 1, len(factor_levels) - 1])]
                span = max_ - min_

                # Shrink the span a bit
                logging.debug('Factor {} span: {}'.format(name, span))
                logging.debug('Factor {}: adjusting span with '
                              'gsd_span_ratio {}'.format(name, span_ratio))
                span = span * span_ratio
                if isinstance(factor, OrdinalFactor) and span < 2.0:
                    # Make sure ordinal factors' spans don't shrink to the
                    # point where there's no spread in the exp. design
                    logging.debug('Factor {}: span ({}) too small, adjusting '
                                  'to minimal span for ordinal factor.'.format(name, span))
                    span = 2.0
                logging.debug('Factor {} span: {}'.format(name, span))

                # center around best point
                best_point = factor_levels[factor_level]
                new_low = best_point - span/2
                new_high = best_point + span/2

                if isinstance(factor, OrdinalFactor):
                    new_low = int(np.round(new_low))
                    new_high = int(np.round(new_high))

                # nudge new high and low so we don't exceed the limits
                new_low, new_high = self._validate_new_factor_limits(
                    factor, name, new_low, new_high)

                # update factors
                factor.current_low = new_low
                factor.current_high = new_high

            optimum_settings[name] = factor_levels[factor_level]
            logging.info('New settings for factor {}:\n{}'.format(
                name, factor))

        results = OptimizationResult(
            pd.Series(optimum_settings),
            converged=False,
            tol=0,
            reached_limits=False,
            empirically_found=True)

        logging.info('Best screening result was exp no {}'.format(optimum_i))
        logging.info('The corresponding response was:\n{}'.format(self._response_values.iloc[optimum_i]))
        if len(self._response_values.columns) > 1:
            logging.info('The combined response was:\n{}'.format(response.iloc[optimum_i]))
        logging.info('The factor settings were:\n{}'.format(results.predicted_optimum))

        # update current best experiment
        self.get_best_experiment(self._design_sheet, self._response_values if len(self._response_values.columns) > 1 else response)
        self._phase = 'optimization'
        return results

    def set_phase(self, phase):
        assert phase in self._allowed_phases, 'phase must be one of {}'.format(self._allowed_phases)
        self._phase = phase

    def _update_numeric_factor(self, factor, name, ratio):
        logging.info('Factor {}: Updating settings.'.format(name))
        logging.info('Factor {}: Current settings: {}'.format(name, factor))
        step_length = self.step_length if self.step_length is not None \
            else abs(ratio)
        step = factor.span * step_length * np.sign(ratio)
        logging.debug('Factor {}: Step by which settings are adjusted is {}.'
            .format(name, step))
        logging.debug('Factor {}: Current span between high and low is {}.'
            .format(name, factor.span))
        logging.debug('Factor {}: Will shrink the span by {}.'
            .format(name, self.shrinkage))
        new_span = factor.span * self.shrinkage
        logging.debug('Factor {}: New span is {}.'.format(name, new_span))
        if isinstance(factor, QuantitativeFactor):
            current_low_new = factor.center + step - new_span / 2
            current_high_new = factor.center + step + new_span / 2
        elif isinstance(factor, OrdinalFactor):
            current_low_new = np.round(factor.center + step - new_span / 2)
            current_high_new = np.round(factor.center + step + new_span / 2)
        else:
            raise NotImplementedError

        # If the proposed step change takes us below or above min and max:
        new_low, new_high = self._validate_new_factor_limits(
            factor, name, current_low_new, current_high_new)

        factor.current_low = new_low
        factor.current_high = new_high
        logging.info('Factor {}: New settings: {}'.format(name, factor))
        logging.info('Factor {}: Done updating.'.format(name))

    def _new_screening_design(self, reduction='auto'):
        factor_items = sorted(self.factors.items())

        levels = list()
        names = list()
        dtypes = list()
        for name, factor in factor_items:
            names.append(name)

            if isinstance(factor, CategoricalFactor):
                levels.append(factor.values)
                dtypes.append(object)
                continue

            num_levels = factor.screening_levels
            spacing = getattr(factor, 'screening_spacing', 'linear')
            min_ = factor.min
            max_ = factor.max
            if not np.isfinite([min_, max_]).all():
                raise ValueError('Can\'t perform screening with unbounded factors')

            space = np.linspace if spacing == 'linear' else np.logspace
            values = space(min_, max_, num_levels)

            if isinstance(factor, OrdinalFactor):
                values = sorted(np.unique(np.round(values)))
                dtypes.append(int)
            else:
                dtypes.append(float)

            levels.append(values)

        design_matrix = pyDOE2.gsd([len(values) for values in levels],
                                   reduction if reduction is not 'auto' else len(levels))
        factor_matrix = list()
        for i, (values, dtype) in enumerate(zip(levels, dtypes)):
            values = np.array(values)[design_matrix[:, i]]
            series = pd.Series(values, dtype=dtype)
            factor_matrix.append(series)

        self._design_matrix = design_matrix
        self._design_sheet = pd.concat(factor_matrix, axis=1, keys=names)
        return self._design_sheet

    def _new_optimization_design(self):
        matrix_designer = self._matrix_designers[self.design_type.lower()]

        numeric_factors = [(name, factor) for name, factor in self.factors.items()
                           if isinstance(factor, NumericFactor)]
        numeric_factor_names = [name for name, factor in numeric_factors]
        design_matrix = matrix_designer(len(numeric_factors))

        mins = np.array([f.min for _, f in numeric_factors])
        maxes = np.array([f.max for _, f in numeric_factors])
        span = np.array([f.span for _, f in numeric_factors])
        centers = np.array([f.center for _, f in numeric_factors])
        factor_matrix = design_matrix * (span / 2.0) + centers

        # Check if current settings are outside allowed design space.
        # Also, for factors that are specified as ordinal, adjust their values
        # in the design matrix to be rounded floats
        for i, (factor_name, factor) in enumerate(numeric_factors):
            if isinstance(factor, OrdinalFactor):
                factor_matrix[:,i] = np.round(factor_matrix[:,i])
            logging.debug('Current setting {}: {}'.format(factor_name, factor))

        if (factor_matrix < mins).any() or (factor_matrix > maxes).any():
            logging.warning(('Out of design space factors. Adjusts factors'
                             'by {}.'.format(self._edge_action + 'ing')))
            if self._edge_action == 'distort':

                # Simply cap out-of-boundary values at mins and maxes.
                capped_mins = np.maximum(factor_matrix, mins)
                capped_mins_and_maxes = np.minimum(capped_mins, maxes)
                factor_matrix = capped_mins_and_maxes

            elif self._edge_action == 'shrink':
                raise NotImplementedError

        factors = list()
        for name, factor in self.factors.items():
            if isinstance(factor, CategoricalFactor):
                values = np.repeat(factor.fixed_value, len(design_matrix))
                factors.append(pd.Series(values))
            else:
                i = numeric_factor_names.index(name)
                dtype = int if isinstance(factor, OrdinalFactor) else float
                factors.append(pd.Series(factor_matrix[:, i].astype(dtype)))

        self._design_sheet = pd.concat(factors, axis=1, keys=self.factors.keys())
        return self._design_sheet

    def _check_convergence(self, centers, converged, criterion, prediction,
                           numeric_factors, recovery=False):
        # It's possible that the optimum is predicted to be at the edge of the allowed
        # min or max factor setting. This will produce a high 'ratio' and the algorithm
        # is not considered to have converged (above). However, in this situation we
        # can't move the space any further and we should stop iterating.

        new_centers = np.array([f.center for f in numeric_factors])
        if (centers == new_centers).all():
            if not recovery:
                logging.info(
                    'The design has not moved since last iteration. Converged.')
            converged = True
            reached_limits = True

            if len(self.responses) > 1 and prediction < 1:
                reached_limits = False
            elif len(self.responses) == 1:
                r_spec = list(self.responses.values())[0]
                low_limit = self._stored_transform(r_spec.get('low_limit', 1))
                high_limit = self._stored_transform(r_spec.get('high_limit', 1))
                if criterion == 'maximize' and 'low_limit' in r_spec:
                    reached_limits = prediction >= low_limit
                elif criterion == 'minimize' and 'high_limit' in r_spec:
                    reached_limits = prediction <= high_limit
                elif criterion == 'target' and 'low_limit' in r_spec and 'high_limit' in r_spec:
                    reached_limits = low_limit <= prediction <= high_limit
        else:
            reached_limits = False
        return converged, reached_limits


def factor_from_spec(f_spec):
    """ Create factor from config factor specification.

    :param dict f_spec: Factor specification from config-file.
    :returns: Function corresponding to `factor_type`.
    :rtype: QuantitativeFactor, OrdinalFactor, CategoricalFactor.
    :raises: UnsupportedFactorType
    """
    factor_type = f_spec.get('type', 'quantitative')
    if factor_type == 'categorical':
        return CategoricalFactor(f_spec['values'])

    elif factor_type == 'quantitative':
        factor_class = QuantitativeFactor
    elif factor_type == 'ordinal':
        factor_class = OrdinalFactor
    else:
        raise UnsupportedFactorType(str(factor_type))

    has_neg = any([f_spec['high_init'] < 0, f_spec['low_init'] < 0])
    f_min = f_spec.get('min', float('-inf') if has_neg else 0)
    f_max = f_spec.get('max', float('inf'))

    factor = factor_class(f_max, f_min, f_spec['low_init'], f_spec['high_init'])
    if 'screening_levels' in f_spec:
        factor.screening_levels = f_spec['screening_levels']

    return factor


def _make_stored_boxcox(lambda_value):
    def boxcox_transform(x):
        return scipy.stats.boxcox(x, lambda_value)
    return boxcox_transform
