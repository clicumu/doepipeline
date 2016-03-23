import abc
import os
import pyDOE
import numpy as np
import pandas as pd
from collections import OrderedDict
try:
    import pymoddeq
except ImportError:
    has_modde = False
else:
    from pymoddeq.utils import FactorType, DesignType, ResponseCriterion
    from pymoddeq.runner import ModdeQRunner
    has_modde = True


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

    def new_design(self):
        """

        :return: Experimental design-sheet.
        :rtype: pandas.DataFrame
        """
        mins = np.array([f.min for f in self.factors.values()])
        maxes = np.array([f.max for f in self.factors.values()])
        span = np.array([f.span for f in self.factors.values()])

        centers = np.array([f.center for f in self.factors.values()]) / 2
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

        return pd.DataFrame(factor_matrix, columns=self.factors.keys())

    def update_factors_from_response(self, response):
        pass


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



if has_modde:
    ModdeQDesigner = _ModdeQDesigner