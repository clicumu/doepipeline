import logging
from itertools import combinations_with_replacement, combinations, chain

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize


class OptimizationFailed(Exception):
    pass


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


def predict_optimum(data_sheet, response, factor_names, criterion='minimize', q2_limit=0.5, **kwargs):
    """
    Fits a model from the experimental sheet and response(s) and returns the
    optimum, model, and prediction. If the Q2 value of the found model is below
    the limit, or if optimization fails, returns None for the optimum and the
    prediction.
    """

    predicted_optimum = None
    optimum = pd.Series([])
    means = data_sheet.mean(axis=0)
    stds = data_sheet.std(axis=0)
    data_sheet = (data_sheet - means) / stds

    x0 = data_sheet.median(axis=0).values

    # Set up bounds for optimization to keep it inside the current design
    # space.
    mins = data_sheet.min(axis=0)
    maxes = data_sheet.max(axis=0)
    bounds = list(zip(mins, maxes))

    data_sheet['_response'] = response

    n_folds = kwargs.get('n_folds', 'loo')
    n_folds = n_folds if n_folds != 'loo' else len(data_sheet)
    model_selection = kwargs.get('model_selection', 'greedy')
    if model_selection == 'greedy':
        model, q2 = stepwise_regression(data_sheet, '_response', n_folds)
    elif model_selection == 'brute':
        model, q2 = brute_force_selection(data_sheet, '_response', n_folds)
    else:
        model = smf.ols(kwargs['manual_formula'], data_sheet).fit()
        q2 = crossvalidate_formula(kwargs['manual_formula'], data_sheet,
                                   '_response', n_folds)

    logging.info('Best model found (Q2={:.4f})'.format(q2))
    logging.info('\n'+str(model.summary()))

    if q2 < q2_limit:
        q2_msg = 'The predictive power of the model (Q2={}) is below the \
        current cutoff ({}). Will not search for an optimum.'.format(
            q2, q2_limit)
        logging.info(q2_msg)
    elif np.isnan(q2):
        q2_msg = 'Failed to create a model. Will not search for an optimum.'
        logging.info(q2_msg)
    else:
        logging.info('Finds the optimum from the current model.')

        # Define optimization function for optimizer.
        def predicted_response(x, invert=False):
            df = pd.DataFrame(np.atleast_2d(x), columns=factor_names)
            return (-1 if invert else 1) * model.predict(df)[0]

        if criterion == 'maximize':
            optimization_results = minimize(
                lambda x: predicted_response(x, True),
                x0, method='L-BFGS-B',
                bounds=bounds)
        elif criterion == 'minimize':
            optimization_results = minimize(
                predicted_response,
                x0,
                method='L-BFGS-B',
                bounds=bounds)

        if not optimization_results['success']:
            logging.info('Was not able to find the optimum: {}'.format(
                optimization_results['message']))
        else:
            predicted_optimum = model.predict(
                pd.DataFrame(optimization_results['x'], index=means.index).T)[0]
            optimum = (optimization_results['x'] * stds) + means
            logging.info('Optimum found:\n{}'.format(optimum))
            logging.info('Predicted response using the found optimum:\n{}'.format(
                predicted_optimum))
    return optimum, model, predicted_optimum


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
    are_quantitative = [np.issubdtype(dtype, np.number) for dtype in data[factor_columns].dtypes]
    factor_columns = [col if is_quantitative else 'C({col})'.format(col=col)
                      for col, is_quantitative in zip(factor_columns, are_quantitative)]

    combs = (combinations(factor_columns, r) for r in range(1, len(factor_columns) + 1))
    factor_combinations = list(chain.from_iterable(combs))

    comb_q2 = list()
    for f_c in factor_combinations:
        q2 = crossvalidate_formula(formula_base + '+'.join(f_c), data, response_column, k)
        comb_q2.append((q2, f_c))
    best_q2, best_combination = sorted(comb_q2)[-1]
    higher_order = ['{}:{}'.format(fac, other_fac) for i, fac in enumerate(best_combination, start=1)
                    for other_fac in best_combination[i:]]
    higher_order += ['np.power({}, 2)'.format(col)
                     for col, is_quant in zip(factor_columns, are_quantitative)
                     if is_quant and col in best_combination]

    while 'still_improving':
        if not higher_order:
            break
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
                        data[factor_columns].dtypes]
    factor_columns = [col if is_quantitative else 'C({col})'.format(col=col)
                      for col, is_quantitative in zip(factor_columns, are_quantitative)]

    higher_order = ['{}:{}'.format(fac, other_fac) for i, fac in
                    enumerate(factor_columns, 1) for other_fac in factor_columns[i:]]
    higher_order += ['np.power({}, 2)'.format(col)
                     for col, is_quant in zip(factor_columns, are_quantitative)
                     if is_quant]
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
