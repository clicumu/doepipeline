from collections import OrderedDict
from itertools import combinations_with_replacement, combinations, chain

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

from doepipeline.designer import OptimizationFailed


def make_desirability_function(response):
    """ Define a Derringer and Suich desirability function.

    :param dict response_dict: Response variable config dictionary.
    :return: desirability function.
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
            elif L <= y <= T:
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

    else:
        raise ValueError(response['criterion'])

    return desirability


def predict_optimum(design_sheet, response, criterion, factors, degree=2):
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

    # Set up bounds for optimization to keep it inside the allowed design space.
    mins = [f.min if f.min != '-inf' else None for f in factors.values()]
    maxes = [f.max if f.max != 'inf' else None for f in factors.values()]
    bounds = list(zip(mins, maxes))

    if criterion == 'maximize':
        optimization_results = minimize(lambda x: predicted_response(x, True),
                                        x0, method='L-BFGS-B', bounds=bounds)
    elif criterion == 'minimize':
        optimization_results = minimize(predicted_response, x0,
                                        method='L-BFGS-B', bounds=bounds)

    if not optimization_results['success']:
        raise OptimizationFailed(optimization_results['message'])

    return optimization_results['x']


def crossvalidate_formula(formula, data, response_column, k):
    PRESS = 0
    for i in range(k):
        start = i * (len(data) // k)
        end = (i + 1) * (len(data) // k) if i < k - 1 else len(data)

        to_drop = data.index[start: end]

        train = data.drop(to_drop)
        test = data.loc[to_drop]

        model = smf.ols(formula, train).fit()
        pred = model.predict(test)
        residuals = test[response_column] - pred
        PRESS += (residuals ** 2).sum()

    response = data[response_column]
    Q2 = 1 - PRESS / ((response - response.mean()) ** 2).sum()
    return Q2


def stepwise_regression(data, response_column, k):
    formula_base = '{} ~ '.format(response_column)
    factor_columns = [col for col in data.columns if col != response_column]
    are_quantitative = [np.issubdtype(dtype, np.number) for dtype in data.dtypes]
    factor_columns = [col if is_quantitative else 'C({col})'.format(col=col)
                      for col, is_quantitative in zip(factor_columns, are_quantitative)]

    combs = (combinations(factor_columns, r) for r in range(1, len(factor_columns) + 1))
    factor_combinations = list(chain.from_iterable(combs))

    comb_q2 = list()
    for f_c in factor_combinations:
        q2 = crossvalidate_formula(formula_base + '+'.join(f_c), data, response_column, k)
        comb_q2.append((q2, f_c))

    best_q2, best_combination = sorted(comb_q2)[-1]
    higher_order = ['{}:{}'.format(fac, other_fac) for i, fac in enumerate(best_combination)
                    for other_fac in best_combination[i:]]

    while 'still_improving':
        term_results = list()
        for term in higher_order:
            terms = [term] + list(best_combination)
            q2 = crossvalidate_formula(formula_base + '+'.join(terms), data, response_column, k)
            term_results.append((q2, term))

        current_best_q2, current_best_term = sorted(term_results)[-1]

        if current_best_q2 > best_q2:
            best_combination = [current_best_term] + list(best_combination)
            higher_order.remove(current_best_term)
            best_q2 = current_best_q2
        else:
            break

    model = smf.ols(formula_base + ' + '.join(best_combination), data).fit()
    return model, best_q2


def brute_force_selection(data, response_column, k):
    formula_base = '{} ~ '.format(response_column)
    factor_columns = [col for col in data.columns if col != response_column]
    are_quantitative = [np.issubdtype(dtype, np.number) for dtype in
                        data.dtypes]
    factor_columns = [col if is_quantitative else 'C({col})'.format(col=col)
                      for col, is_quantitative in zip(factor_columns, are_quantitative)]

    higher_order = ['{}:{}'.format(fac, other_fac) for i, fac in
                    enumerate(factor_columns) for other_fac in factor_columns[i:]]
    all_factors = factor_columns + higher_order
    combs = (combinations(all_factors, r) for r in range(1, len(all_factors) + 1))
    factor_combinations = list(chain.from_iterable(combs))

    comb_q2 = list()
    for f_c in factor_combinations:
        q2 = crossvalidate_formula(formula_base + '+'.join(f_c), data,
                                   response_column, k)
        comb_q2.append((q2, f_c))

    best_q2, best_combination = sorted(comb_q2)[-1]
    model = smf.ols(formula_base + ' + '.join(best_combination), data).fit()
    return model, best_q2