import abc
import os
import pyDOE
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict
from itertools import combinations_with_replacement
try:
    import pymoddeq
except ImportError:
    has_modde = False
else:
    from pymoddeq.utils import FactorType, DesignType, ResponseCriterion
    from pymoddeq.runner import ModdeQRunner
    has_modde = True


class OptimizationFailed(Exception):
    pass


class Factor:

    def __init__(self, factor_max, factor_min, factor_type):
        self.max = factor_max
        self.min = factor_min
        self.type = factor_type
        self.current_high = None
        self.current_low = None

    @property
    def span(self):
        return self.current_high - self.current_low

    @property
    def center(self):
        return (self.current_high + self.current_low) / 2


class UnsupportedDesign(Exception):
    pass


class BaseExperimentDesigner:

    __metaclass__ = abc.ABCMeta

    def __init__(self, factors, design_type, responses, at_edges='distort'):
        try:
            assert at_edges in ('distort', 'shrink'),\
                'unknown action at_edges: {0}'.format(at_edges)
        except AssertionError as e:
            raise ValueError(str(e))

        self.factors = OrderedDict()
        for factor_name, f_spec in factors.items():
            has_neg = any([f_spec['high_init'] < 0, f_spec['low_init'] < 0])
            f_min = f_spec.get('min', float('-inf') if has_neg else 0)
            f_max = f_spec.get('max', float('inf'))
            f_type = f_spec.get('type', 'quantitative')
            factor = Factor(f_max, f_min, f_type)
            factor.current_high = f_spec['high_init']
            factor.current_low = f_spec['low_init']
            self.factors[factor_name] = factor

        self.design_type = design_type
        self.responses = responses
        self._edge_action = at_edges

    @abc.abstractmethod
    def new_design(self):
        pass

    @abc.abstractmethod
    def update_factors_from_response(self, response):
        pass


class ExperimentDesigner(BaseExperimentDesigner):

    _matrix_designers = {
        'fullfactorial2levels': pyDOE.ff2n,
        'fullfactorial3levels': lambda n: pyDOE.fullfact([3] * n),
        'placketburman': pyDOE.pbdesign,
        'boxbehnken': lambda n: pyDOE.bbdesign(n, 1),
        'ccc': lambda n: pyDOE.ccdesign(n, (0, 1), face='ccc'),
        'ccf': lambda n: pyDOE.ccdesign(n, (0, 1), face='ccf'),
        'cci': lambda n: pyDOE.ccdesign(n, (0, 1), face='cci'),
    }

    def __init__(self, *args, **kwargs):
        super(ExperimentDesigner, self).__init__(*args, **kwargs)
        n = len(self.factors)
        try:
            matrix_designer = self._matrix_designers[self.design_type.lower()]
        except KeyError:
            raise UnsupportedDesign(self.design_type)
        else:
            self._design_matrix = matrix_designer(n)

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
        mins = np.array([f.min for f in self.factors.values()])
        maxes = np.array([f.max for f in self.factors.values()])
        span = np.array([f.span for f in self.factors.values()])

        centers = np.array([f.center for f in self.factors.values()])
        factor_matrix = self._design_matrix * (span / 2) + centers

        # Check if current settings are outside allowed design space.
        if (factor_matrix < mins).any() or (factor_matrix > maxes).any():
            if self._edge_action == 'distort':

                # Simply cap out-of-boundary values at mins and maxes.
                capped_mins = np.maximum(factor_matrix, mins)
                capped_mins_and_maxes = np.minimum(capped_mins, maxes)
                factor_matrix = capped_mins_and_maxes

            elif self._edge_action == 'shrink':
                raise NotImplementedError

        self._design_sheet = pd.DataFrame(factor_matrix, columns=self.factors.keys())
        return self._design_sheet

    def update_factors_from_response(self, response, degree=2):
        """ Calculate optimal factor settings given response and update
        factor settings to center around optimum.

        If several responses are defined, the geometric mean of Derringer
        and Suich's desirability functions will be used for optimization,
        see:

        Derringer, G., and Suich, R., (1980), "Simultaneous Optimization
        of Several Response Variables," Journal of Quality Technology, 12,
        4, 214-219.

        :param pandas.DataFrame response: Response sheet.
        :param int degree: Degree of polynomial to fit.
        """
        if response.shape[1] == 1:
            criterion = list(self.responses.values())[0]['criterion']
        else:
            raise NotImplementedError

        # Find predicted optimal factor setting.
        optimal_x = predict_optimum(self._design_sheet, response,
                                    criterion, degree)

        # Update factors around predicted optimal settings, but keep
        # the same span as previously.
        spans = np.array([f.span for f in self.factors.values()])
        new_highs = optimal_x + spans / 2
        new_lows = optimal_x - spans / 2
        for i, key in enumerate(self._design_sheet.columns):
            self.factors[key].current_high = new_highs[i]
            self.factors[key].current_low = new_lows[i]

        return pd.Series(optimal_x, self._design_sheet.columns)


class _ModdeQDesigner(BaseExperimentDesigner):

    def __init__(self, factors, design_type, responses):
        super(_ModdeQDesigner, self).__init__(factors, design_type, responses)
        self.modde = None

        # These mappings are added as instance attributes rather than class-
        # attributes to avoid NameError which occurs of pymoddeq isn't
        # installed.
        self._modde_factor_types = {
            'undefined': FactorType.NotDefinedFactorType,
            'quantitative': FactorType.Quantitative,
            'quantitative_multilevel': FactorType.QuantitativeMultiLevel,
            'qualitative': FactorType.Qualitative,
            'formulation': FactorType.Formulation,
            'filler': FactorType.Filler
        }
        self._modde_response_criterions = {
            'undefined': ResponseCriterion.Undefined,
            'minimize': ResponseCriterion.Minimize,
            'maximize': ResponseCriterion.Maximize,
            'target': ResponseCriterion.Target,
            'exclude': ResponseCriterion.Exclude,
            'predict': ResponseCriterion.Predict
        }
        self._modde_design_types = {
            'notdefineddesigntype': DesignType.NotDefinedDesignType,
            'fullfactorial2levels': DesignType.FullFactorial2Levels,
            'fullfactorial3levels': DesignType.FullFactorial3Levels,
            'fullfactorialmixed': DesignType.FullFactorialMixed,
            'fractionalfactorialresolution3': DesignType.FractionalFactorialResolution3,
            'fractionalfactorialresolution4': DesignType.FractionalFactorialResolution4,
            'fractionalfactorialresolution5': DesignType.FractionalFactorialResolution5,
            'doptimal': DesignType.DOptimal,
            'rechtschaffner': DesignType.Rechtschaffner,
            'fractional9': DesignType.Fractional9,
            'fractional18': DesignType.Fractional18,
            'fractional27': DesignType.Fractional27,
            'fractional36': DesignType.Fractional36,
            'placketburman': DesignType.PlacketBurman,
            'boxbehnken': DesignType.BoxBehnken,
            'ccc': DesignType.CCC,
            'ccf': DesignType.CCF,
            'cco': DesignType.CCO,
            'reducedccc': DesignType.ReducedCCC,
            'reducedccf': DesignType.ReducedCCF,
            'axialnormal': DesignType.AxialNormal,
            'axialextended': DesignType.AxialExtended,
            'axialreduced': DesignType.AxialReduced,
            'simplexmodified': DesignType.SimplexModified,
            'simplexface': DesignType.SimplexFace,
            'simplexspecialcubic': DesignType.SimplexSpecialCubic,
            'simplexcubic': DesignType.SimplexCubic,
            'doehlert': DesignType.Doehlert,
            'redmup': DesignType.REDMup,
            'onion': DesignType.Onion,
            'levels': DesignType.Levels,
            'plugin': DesignType.Plugin,
            'pb_ss': DesignType.PB_SS,
            'custom': DesignType.Custom,
        }

    def connect_to_modde(self, oem_string,
                         investigation_name='investigation.mip',
                         workdir='.',
                         process_model_type=3,
                         mixture_model_type=0, **kwargs):
        modde_config = {
            'oemString': oem_string,
            'investigationName': os.path.abspath(investigation_name),
            'investigationFolder': workdir,
            'processModelType': process_model_type,
            'mixtureModelType': mixture_model_type,
            'designType': self._modde_design_types[self.design_type.lower()]
        }
        modde_config.update(kwargs)

        modde_factors = dict()
        for i, (factor, spec) in enumerate(self.factors.items(), start=1):
            # Provide defaults required by MODDE-bindings but not
            # specified.
            factor_type = spec['type'].lower()\
                if 'type' in spec else 'quantitative'
            factor_min = spec['min'] if 'min' in factor else float('-inf')
            factor_max = spec['max'] if 'max' in factor else float('inf')

            modde_factor = {
                'name': factor,
                'abbr': 'Fac{0}'.format(i),
                'type': self._modde_factor_types[factor_type],
                'min': factor_min,
                'max': factor_max,
                'low_limit': spec['low_init'],
                'high_limit': spec['high_init']
            }
            modde_factors[factor] = modde_factor

        modde_config['factors'] = modde_factors

        modde_responses = dict()
        for i, (response, criterion) in enumerate(self.responses.items(), start=1):
            modde_response = {
                'name': response,
                'abbr': 'Res{0}'.format(i),
                'type': 1,  # Need to look if if sensible to allow other.
                'criterion': self._modde_response_criterions[criterion]
            }
            modde_responses[response] = modde_response

        modde_config['responses'] = modde_responses

        # If connection fails, let exception bubble.
        self.modde = ModdeQRunner(modde_config)

    def new_design(self):
        self._design = self.modde.get_experimental_setup()

    def update_factors_from_response(self, response):
        pass


def make_desirability_function(response):
    """ Define a Derringer and Suich desirability function.

    :param response_dict:
    :return:
    :rtype: Callable
    """
    s = response.get('priority', 1)
    if response['criterion'] == 'target':
        L = response['low_limit']
        U = response['high_limit']
        T = response.get('target', (U + L) / 2)

        def desirability(y):
            if y < L or U < y:
                return 0
            elif  L <= y <= T:
                return ((y - L) / (T - L)) ** s
            elif T <= y <= U:
                return ((y - U) / (T - U)) ** s

    elif response['criterion'] == 'maximize':
        L = response['low_limit']
        T = response['target']

        def desirability(y):
            if y < L:
                return 0
            elif T < y:
                return 1
            else:
                return ((y - L) / (T - L)) ** s

    elif response['criterion'] == 'minimize':
        U = response['high_limit']
        T = response['target']

        def desirability(y):
            if y < T:
                return 1
            elif U < y:
                return 0
            else:
                return ((y - U) / (T - U)) ** s

    return desirability


def predict_optimum(design_sheet, response, criterion, degree=2):
        """ Regress using `degree`-polynomial and optimize response.

        :param pandas.DataFrame design_sheet: Factor settings.
        :param pandas.Series response: Response Y-vector.
        :param str criterion: 'maximize' | 'minimize'
        :param int degree: Degree of polynomial to use for regression.
        :return: Array with predicted optimum.
        :rtype: numpy.ndarray[float]
        :raises: OptimizationFailed
        """
        matrix_columns = design_sheet.columns
        n_rows, n_cols = design_sheet.shape
        products = OrderedDict()
        combinations = [list(comb) for deg in range(1, degree + 1)
                        for comb in combinations_with_replacement(range(n_cols), deg)]

        # Multiply columns together.
        for column_group in (matrix_columns[comb] for comb in combinations):
            values = design_sheet[column_group].product(axis=1)
            products['*'.join(column_group)] = values

        extended_sheet = pd.DataFrame(products)

        # Extend data-sheet with constants-column.
        design_w_constants = np.hstack([np.ones((n_rows, 1)),
                                        extended_sheet.values])

        # Regress using least squares.
        c_stacked = np.linalg.lstsq(design_w_constants, response.values)[0]
        coefficients = c_stacked.flatten()

        # Define optimization function for optimizer.
        def predicted_response(x, invert=False):
            # Make extended factor list from given X.
            factor_list = [1] + [np.product(x[comb]) for comb in combinations]
            factors = np.array(factor_list)

            # Calculate response.
            factor_contributions = np.multiply(factors, coefficients)
            return (-1 if invert else 1) * factor_contributions.sum()

        # Since factors are set according to design the optimum is already
        # guessed to be at the center of the design. Hence, use medians as
        # initial guess.
        x0 = design_sheet.median(axis=0).values

        if criterion == 'maximize':
            optimization_results = minimize(lambda x: predicted_response(x, True), x0)
        elif criterion == 'minimize':
            optimization_results = minimize(predicted_response, x0)

        if not optimization_results['success']:
            raise OptimizationFailed(optimization_results['message'])

        return optimization_results['x']


if has_modde:
    ModdeQDesigner = _ModdeQDesigner