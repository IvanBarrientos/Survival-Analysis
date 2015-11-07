# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:43:32 2015

Purpose
-------
Basic Aalen additive model.

@author: ibarrien
"""



'''
Notes on Aalen additive
-----------------------
n = num train samples
p = num features, get p + 1 bc add 1's

X_k = updated matrix at time k based on y/n cancel at k-1, shape = n, p+1
B_k = (X'_k.X_k)^-1 . X'_k.y_k, shape = p+1, 1
B(t)  = B_0 + ... + B_t, cumulative sum of B_k's, shape = p+1, 1

Note: we set B(t)=: t_B

x_test: test row, shape = 1, p+1
H(x_test, B(t)) = x_test . B(t), shape = 1, 1
P(x_test, t) = exp(-H(x_test, B(t))), shape = 1,1 
P(x_test, t) is prob model that x_test stays for at least t total days


'''

import copy
import numpy as np
import pandas as pd
from numpy import exp
from math import factorial
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression as LR


class AalenAdditiveModel():

    """
    This class fits the regression model:

    hazard(t)  = b_0(t) + b_t(t)*x_1 + ... + b_N(t)*x_N

    The hazard rate is a linear function of the covariates.

    """

    def __init__(self, fit_intercept=True, zero_bump=False):
        self.fit_intercept = fit_intercept
        self.zero_bump = zero_bump


    def fit(self, df_train, features, duration_var, event_var):
        '''Train the model. 
        
        Params
        ------
        df_train (DataFrame): data to train on
        features (array(object)): list of features i.e. covariates
        duration_var (object): variable name for set of times
        event_var (object): variable name for set of cancel/did not cancel
        
        Notes
        -----
        This is where the majority of initialization takes place.
        '''
        # preprocessing        
        df_train.sort(duration_var, inplace=True)
        events = df_train[event_var]
        
        
        '''INITIALIZE ATTRIBUTES'''        
        # add properties
        self.num_samples = df_train.shape[0]
        self.features = features
        # set event vals as binary wrt respsonse r
        try:
            self.events = np.array([0 if r=='N' else 1 for r in events])
        except:
            self.events = np.array(events)
  
        ## set timeline
        timeline = list(df_train[duration_var])
        timeline.sort(reverse=False)  # sort in ascending order by duration val
        self.timeline = timeline
        
        ## format covariates to train
        self.df_ftrs = np.c_[np.ones(self.num_samples), df_train[features]]  # append 1st as 1st col for intercept term
        self.num_cols = self.df_ftrs.shape[1]
        
        
        ## GET X_k, B_k matrices -> part of self
        X_list = []
        # H_function = []
        
        ### initialize X with val
        X_0 = self.df_ftrs.copy()
        X_list.append(X_0)
        self.X_list = X_list
        
        ### initialize B_function, viewed as function time ind -> features
        B_function = []  
        self.B_function = B_function
        
        ### initalize list of partial summands for B_function
        B_list =[]
        self.B_list = B_list
        
        '''GET VALUES FOR ALL ATTRIBUTES'''
        # get X's
        for k in xrange(1, self.num_samples):
            #prev_cancel = self.events[k-1]==1  # setup a cancel column
            new_X = self.get_X_k(k)
            self.X_list.append(new_X)
            
        # get partial summands of B(t)
        for k in xrange(0, self.num_samples):
            #curr_canceled = event_events[k]==1 
            B = self.get_B_k(k)
            self.B_list.append(B)
        
        # get cumulative sums, i.e. B(t) function
        for k in xrange(0, self.num_samples):
            if k==0:
                self.B_function.append(B_list[0])
            else:
                self.B_function.append(B_function[k-1] + B_list[k])
        
        # return    

    '''TRAIN UTILS'''
    
    def get_X_k(self, curr_row):
        '''Get updated X matrix at time t=k.
        
        Params
        -------
        curr_row (int): = k, row index at time t=k
        
        Returns
        -------
        dx (DataFrame): updated at time t=k version of dx 
        
        Notes
        -----
        prev_X should not have a cancel column
        indexing of prev_X starts at 0
        '''
        prev_X = self.X_list[curr_row-1]
        new_X = copy.deepcopy(prev_X)  # need deepcopy, else changes prev_X!
        if curr_row <= 0:
            #logger 'got curr row=', curr_row  # logging
            return new_X
        elif self.events[curr_row-1]:
            '''update step: set prev row = 0 if prev row canceled'''
           # logger 'encountered cancel at step', curr_row
            new_X[curr_row - 1] = 0
        return new_X  # store at a kth place in self
        
    def get_B_k(self, curr_row):
        '''Get the t=k part of B(t):=t_B, i.e. k-th summand of B(t).
        
        Params
        ------
        curr_row (int): row index at time t=k
        
        Returns
        -------
        res (DataFrame) Partial k-th summand of B(t).
        
        Notes
        -----
        The returned res is used in forming the sum B(t)
        '''
        X_k = self.X_list[curr_row]    
        
        # if curr row did not cancel, return zeros array
        if not self.events[curr_row]:
            if not self.zero_bump:
                res = np.zeros(self.num_cols)
            else:
                res = np.array([0.001]*self.num_samples)
        # if curr row did cancel, find ols 
        else:
            y = np.zeros(self.num_samples)
            y[curr_row] = 1
            lr = LR()
            lr.fit(X_k, y)
            res = lr.coef_
            '''
            # Alternatively:
            y = np.zeros(self.num_samples)
            y[curr_row] = 1
            a = inv(np.dot(X_k.T, X_k))
            b = np.dot(X_k.T, y)
            res = np.dot(a,b)
            '''
        return res

        
    '''PREDICT UTILS'''
    
    def hazards_function(self):
        '''Get Hazards function for test sample test_row
        
        Returns
        -------
        (array) hazard function for x, len(x) = self.num_samples
        
        Notes
        -----
        self contains (formatted) test_row attribute at this point
        '''
        H_function=[]
        for k in xrange(0, self.num_samples):
            if k==0:
                H_function.append(np.dot(self.test_row, self.B_function[0]))
            else:
                H_function.append(H_function[k-1] + np.dot(self.test_row, self.B_function[k]))  # caution!
        return np.array(H_function)
        
    def survival_function(self, exp_n=0):
        '''Get survival function as exp(-H), with exp truncation optional.
        
        Params
        ------
        exp_n (int): nonneg integer for Taylor expansion order for exp approx.
        
        Returns
        -------
        (array) survival function for x, len(x) = self.num_samples
                
        Notes
        -----
        self contains (formatted) test_row attribute at this point
        '''
        if exp_n <= 1:
            return exp(-self.hazards_function())
        else:
            y = -self.hazards_function()
            res = 0
            for k in xrange(0, exp_n):
                res += y**k / float(factorial(k))
            return res      
        
    def predict(self, x):
        '''Predict as area under survival curve for test_row.
        
        Params
        ------
        x (float/array): test data with #features-many values
        
        Returns
        -------
        (float) Area under survival curve for x using trapezoid rule.
        
        Notes
        -----
        self attribute test_row set here
        values of x should be in order of features list
        IN PROGRESS: check for repeated vals in self.timeline and take avgs
        '''
        
        test_row = [1]  # initialize with 1 in front if intercept true
        try:
            test_row.extend(x)
        except:
            test_row.append(x)  # single feature case
        self.test_row = test_row
        return trapz(self.survival_function(), self.timeline)
        
        

                
    def total_expectation(self, x):
        '''Make prediction for x's lifetime as Expectation value'''
        test_row = [1]  # initialize with 1 in front if intercept true
        try:
            test_row.extend(x)
        except:
            test_row.append(x)  # single feature case
        self.test_row = test_row
        return np.dot(self.survival_function(), self.timeline)            
        
    def hazard_t(x, k_B):
        '''Hazard 'rate' at time t=k.
        
        Params
        -------
        x: test row
        k_B: matrix B(t=k), equal to sum of B_i's for i=0,.., k
        '''
        return np.dot(x, k_B)
      
            
        
    
        
    
    
