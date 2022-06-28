""" Class svm containing definitions for Chapter TMA 
by Simon van Hemert
date created: 07-07-21 """

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
import random
import pandas as pd
import numpy as np

""" Standard scatter plot and regression line """
def plot_reg(axes, x, y, model, x_lab="x", y_lab="y", title="Linear Regression"):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    x: (single) Feature
    y: Result
    model: fitted linear sm model  """
    # Plot scatter data
    sns.regplot(x=x, y=y, 
                scatter=True, ci=False, lowess=False, 
                scatter_kws={'s': 40, 'alpha': 0.5},
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    # Set labels:
    axes.set_xlabel(x_lab)
    axes.set_ylabel(y_lab)
    axes.set_title(title)


""" Plot Residuals vs Fitted Values """
def plot_residuals(axes, yfit, res, n_samp=0, 
                   x_lab='Fitted Values', y_lab='Residuals', title='Residuals vs Fitted'):
    """ Inputs: 
    axes: axes created with matplotlib.pyplot
    x: x values
    yfit: fitted/predicted y values 
    n_samp[optional]: number of resamples"""
    # For every random resampling
    for i in range(n_samp):
        # 1. resample indices from Residuals
        samp_res_id = random.sample(list(res), len(res))
        # 2. Average of Residuals, smoothed using LOWESS 
        sns.regplot(x=yfit, y=samp_res_id, 
                    scatter=False, ci=False, lowess=True, 
                    line_kws={'color': 'lightgrey', 'lw': 1, 'alpha': 0.8})
        # 3. Repeat again for n_samples 
        
    df = pd.concat([yfit, res], axis=1)
    axes = sns.residplot(x=yfit, y=res, data=df, 
                    lowess=True, scatter_kws={'alpha': 0.5}, 
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    axes.set_title(title)
    axes.set_ylabel(y_lab)
    axes.set_xlabel(x_lab)
    
    
""" Scale-Location Plot """
def plot_scale_loc(axes, yfit, res_stand_sqrt, n_samp=0, 
                   x_lab='Fitted Values', y_lab='$\sqrt{\|Standardized\ Residuals\|}$',
                   title='Scale-Location plot'):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    yfit: fitted/predicted y values
    res_stand_sqrt: Absolute square root Residuals
    n_samp[optional]: number of resamples """
    # For every random resampling
    for i in range(n_samp):
        # 1. resample indices from sqrt Residuals
        samp_res_id = random.sample(list(res_stand_sqrt), len(res_stand_sqrt))
        # 2. Average of Residuals, smoothed using LOWESS 
        sns.regplot(x=yfit, y=samp_res_id, 
                    scatter=False, ci=False, lowess=True, 
                    line_kws={'color': 'lightgrey', 'lw': 1, 'alpha': 0.8})
        # 3. Repeat again for n_samples

    # plot Regression usung Seaborn
    sns.regplot(x=yfit, y=res_stand_sqrt, 
                scatter=True, ci=False, lowess=True, 
                scatter_kws={'s': 40, 'alpha': 0.5},
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    axes.set_title(title)
    axes.set_ylabel(y_lab)
    axes.set_xlabel(x_lab)

    
""" QQ Plot standardized residuals """
def plot_QQ(axes, res_standard, n_samp=0, 
            x_lab='Theoretical Quantiles', y_lab='Standardized Residuals', 
            title='Normal Q-Q'):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    res_standard: standardized residuals 
    n_samp[optional]: number of resamples """
    # QQ plot instance
    QQ = ProbPlot(res_standard)
    # Split the QQ instance in the seperate data
    qqx = pd.Series(sorted(QQ.theoretical_quantiles), name="x")
    qqy = pd.Series(QQ.sorted_data, name="y")
    if n_samp != 0:
        # Estimate the mean and standard deviation
        mu = np.mean(qqy)
        sigma = np.std(qqy)
        # For ever random resampling
        for lp in range(n_samp):
            # Resample indices 
            samp_res_id = np.random.normal(mu, sigma, len(qqx))
            # Plot
            sns.regplot(x=qqx, y=sorted(samp_res_id),
                        scatter=False, ci=False, lowess=True, 
                        line_kws={'color': 'lightgrey', 'lw': 1, 'alpha': 0.8})

    sns.regplot(x=qqx, y=qqy, scatter=True, lowess=False, ci=False,
                scatter_kws={'s': 40, 'alpha': 0.5},
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    axes.plot(qqx, qqx, '--k', alpha=0.5)
    axes.set_title(title)
    axes.set_ylabel(y_lab)
    axes.set_xlabel(x_lab)
    
    
""" Cook's distance """
def plot_cooks(axes, res_inf_leverage, res_standard, n_pred=1, 
               x_lim=None, y_lim=None, n_levels=4):
    """ Inputs:
    axes: axes created with matplotlib.pyplot
    res_inf_leverage: Leverage
    res_standard: standardized residuals
    n_pred: number of predictor variables in x
    x_lim, y_lim[optional]: axis limits
    n_levels: number of levels"""
    sns.regplot(x=res_inf_leverage, y=res_standard, 
                scatter=True, ci=False, lowess=True, 
                scatter_kws={'s': 40, 'alpha': 0.5},
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    # Set limits
    if x_lim != None:
        x_min, x_max = x_lim[0], x_lim[1]
    else:
        x_min, x_max = min(res_inf_leverage), max(res_inf_leverage) 
    if y_lim != None:
        y_min, y_max = y_lim[0], y_lim[1]
    else:
        y_min, y_max = min(res_standard), max(res_standard) 
    
    # Plot centre line
    plt.plot((x_min, x_max), (0, 0), 'g--', alpha=0.8)
    # Plot contour lines for Cook's Distance levels
    n = 100
    cooks_distance = np.zeros((n, n))
    x_cooks = np.linspace(x_min, x_max, n)
    y_cooks = np.linspace(y_min, y_max, n)
    
    for xi in range(n):
        for yi in range(n):
            cooks_distance[yi][xi] = \
            y_cooks[yi]**2 * x_cooks[xi] / (1 - x_cooks[xi]) / (n_pred + 1)
    CS = axes.contour(x_cooks, y_cooks, cooks_distance, levels=n_levels, alpha=0.6)

    axes.clabel(CS, inline=0,  fontsize=10)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_title('Residuals vs Leverage and Cook\'s distance')
    axes.set_xlabel('Leverage')
    axes.set_ylabel('Standardized Residuals')
    
    
