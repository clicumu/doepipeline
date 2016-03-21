import abc
import os

try:
    import pymoddeq
except ImportError:
    has_modde = False
else:
    from pymoddeq.utils import FactorType, DesignType, ResponseCriterion
    from pymoddeq.runner import ModdeQRunner
    has_modde = True


class BaseExperimentDesigner:

    __metaclass__ = abc.ABCMeta

    def __init__(self, factors, design_type, responses):
        self.factors = factors
        self.design_type = design_type
        self.responses = responses

    @abc.abstractmethod
    def new_design(self, factor_settings=None):
        pass

    @abc.abstractmethod
    def new_design_from_response(self, response):
        pass


class ExperimentDesigner(BaseExperimentDesigner):
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

    def new_design(self, factor_settings=None):
        self._design = self.modde.get_experimental_setup()

    def new_design_from_response(self, response):
        pass



if has_modde:
    ModdeQDesigner = _ModdeQDesigner