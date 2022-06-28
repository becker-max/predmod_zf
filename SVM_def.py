""" Class svm containing definitions for Chapter SVM 
by Simon van Hemert
date created: 09-10-20
data updated: 09-10-20"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class SVM_def:
    def __init__(self):
        self.xx, self.yy = [], []
        pass

    def table_scores(self, ypredicted, ytrue):
        """ Return table showing predicted and true scores in n*n matrix 
        Inputs: 
        - Vector containing n predicted values 
        - Vector containing n true values
        Returns: 
        n*n Matrix with number of correct predictions on diagonal """
        # Empty Matrix:
        lables = np.unique(ytrue, return_inverse=False, axis=None)
        n = len(lables)
        scores = np.zeros((n, n))
        # Fill matrix with values:
        for i in range(len(ytrue)):
            true_class, pred_class = int(ytrue[i]), int(ypredicted[i])
            scores[np.where(true_class == lables)[0][0]][
                np.where(pred_class == lables)[0][0]] += 1
        # Name rows and columns:
        r, c = [], []
        for i in range(len(lables)):
            r.append(("True " + str(lables[i])))
            c.append(("Pred " + str(lables[i])))
        scores = pd.DataFrame(scores, columns=c, index=r)
        
        return scores

    def create_grid(self, x, n, margin=0.5):
        """ Create a grid for x,
        where x = i*2 dimensional array
        n = number of points per axis
        margin = margin around extreme values."""
        xx = np.linspace(min(x[:, 0]) - margin, max(x[:, 0]) + margin, n)
        yy = np.linspace(min(x[:, 1]) - margin, max(x[:, 1]) + margin, n)
        self.yy, self.xx = np.meshgrid(yy, xx)
        xy = np.vstack([self.xx.ravel(), self.yy.ravel()]).T

        return xy, self.xx, self.yy

    def svm_plot(self, axis, x, y, Z, clf, xtest=None, ytest=None, 
                 plottest=False, coloring=False, suppvector=True):
        """ Scatter plot of x1 x2 and y
        x[n, 2] contains both axis
        y[n] contains class info
        Z[n][n] contains decision function
        clf contains fit SVM
        axis is the axis to plot on
        plottest also plots testdate as x
        Coloring also colors disicion surface based on Z an plane 0"""
        # Coloring, when applicable
        if coloring:
            colors = np.where(Z < 0, 0, 1)
            # If Dimensions of Colors(Z) are the same as x, 
            # drop last row and column for shading
            if colors.shape[1] == len(self.xx): 
                colors = np.delete(colors, [-1], axis=0)
                colors = np.delete(colors, [-1], axis=1)
            plt.pcolor(self.xx, self.yy, colors, cmap=plt.cm.coolwarm, 
                       alpha=0.2, edgecolors='face')
        # Data points
        axis.scatter(x[:, 0], x[:, 1], c=y, cmap=cm.coolwarm, marker="o")
        # Test data points, when applicable
        if plottest:
            axis.scatter(xtest[:, 0], xtest[:, 1], c=ytest, 
                         cmap=cm.coolwarm, marker="x", alpha=0.5)
            
        # Division line
        axis.contour(self.xx, self.yy, Z, colors='k', levels=[-1, 0, 1], 
                     alpha=0.5,
                   linestyles=['--', '-', '--'])
        # Support Vectors
        if suppvector:
            axis.scatter(clf.support_vectors_[:, 0], 
                         clf.support_vectors_[:, 1], s=100,
                         linewidth=1, facecolors='none', edgecolors='k')
        # Labels
        plt.xlabel("X1"), plt.ylabel("X2")



