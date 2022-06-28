""" Definitions writen and used in Chapter Linear Model Selection """
import pandas as pd
import numpy as np
import statsmodels.api as sm


def fit_linear_reg(x, y):
    '''Fit Linear model with predictors x on y 
    return AIC, BIC, R2 and R2 adjusted '''
    x = sm.add_constant(x)
    # Create and fit model
    model_k = sm.OLS(y, x).fit()
    
    # Find scores
    BIC = model_k.bic
    AIC = model_k.aic
    R2 = model_k.rsquared
    R2_adj = model_k.rsquared_adj
    RSS = model_k.ssr
    
    # Return result in Series
    results = pd.Series(data={'BIC': BIC, 'AIC': AIC, 'R2': R2,
                              'R2_adj': R2_adj, 'RSS': RSS})
    
    return results


def add_one(x_full, x, y, scoreby='RSS'):
    ''' Add possible predictors from x_full to x, 
    Fit a linear model on y using fit_linear_reg
    Returns Dataframe showing scores as well as best model '''
    # Predefine DataFrame
    x_labels = x_full.columns
    zeros = np.zeros(len(x_labels))
    results = pd.DataFrame(
        data={'Predictor': x_labels.values, 'BIC': zeros, 
               'AIC': zeros, 'R2': zeros, 
               'R2_adj': zeros, 'RSS': zeros})

    # For every predictor find R^2, RSS, and AIC
    for i in range(len(x_labels)):
        x_i = np.concatenate((x, [np.array(x_full[x_labels[i]])]))
        results.iloc[i, 1:] = fit_linear_reg(x_i.T, y)
        
    # Depending on where we scoreby, we select the highest or lowest
    if scoreby in ['RSS', 'AIC', 'BIC']:
        best = x_labels[results[scoreby].argmin()]
    elif scoreby in ['R2', 'R2_adj']:
        best = x_labels[results[scoreby].argmax()]
        
    return results, best 

def drop_one(x, y, scoreby='RSS'):
    ''' Remove possible predictors from x, 
    Fit a linear model on y using fit_linear_reg
    Returns Dataframe showing scores as well as predictor 
    to drop in order to keep the best model '''
    # Predefine DataFrame
    x_labels = x.columns
    zeros = np.zeros(len(x_labels))
    results = pd.DataFrame(
        data={'Predictor': x_labels.values, 'BIC': zeros, 
               'AIC': zeros, 'R2': zeros, 
               'R2_adj': zeros, 'RSS': zeros})

    # For every predictor find RSS and R^2
    for i in range(len(x_labels)):
        x_i = x.drop(columns=x_labels[i])
        results.iloc[i, 1:] = fit_linear_reg(x_i, y)
    
    # Depending on where we scoreby, we select the highest or lowest
    if scoreby in ['RSS', 'AIC', 'BIC']:
        worst = x_labels[results[scoreby].argmin()]
    elif scoreby in ['R2', 'R2_adj']:
        worst = x_labels[results[scoreby].argmax()]
    
    return results, worst 

