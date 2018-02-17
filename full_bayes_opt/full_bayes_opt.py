#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (YYYY-MM-DD)

"""
Bayesian optimization of black-box functions using a fully Bayesian surrogate model to marginalize over
alternative hyperparameter choices.
"""

from __future__ import division

import os


class BayesianOptimizer(object):
  def __init__(self, objective_functon, bounding_box, acquisition_function, stan_surrogate_model_path):
    """

    :param objective_functon: function object - the function under minimization; must return a float
    :param bounding_box: iterable containing the max and min for each dimension. Order of max and min is irrelevant,
     but all outputs depend on the order in which the dimensions are supplied.
     For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
       [(2,-2),(-5,5)] or
       [[-2,2],[5,-5]] or
       np.array([[2,-2],[5,-5])
     or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
     Note that all computations are done on the scale of the supplied parameters. If you want to do something like fit
     a parameter that varies on the log scale, supply log-scaled coordinates and then do the appropriate
     back-transformation in the call to the objective_function.
    :param acquisition_function: string - string must be in {"PI", "EQI", "EI", "UCB"}
      PI - probability of improvement
      EQI - expected quantile improvement
      EI - expected improvement
      UCB - upper confidence bound
    :param stan_surrogate_model_path:
    """
    self._obj_fn = objective_functon
    self.bounding_box = bounding_box
    self.acquisition_function = acquisition_function

    try:
      with open(stan_surrogate_model_path) as f:
        self._model_str = f.read()
    except FileNotFoundError:
      load_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), stan_surrogate_model_path)
      with open(load_path) as f:
        self._model_str = f.read()

  @property
  def obj_fn(self):
    return self._obj_fn

  @property
  def model_str(self):
    return self._model_str

  def forward_transform(self):
    return

  def back_transform(self):
    return

  def PI(self):
    return

  def EQI(self):
    return

  def EI(self):
    return

  def UCB(self):
    return

  def fit(self):
    return

  def get_next(self):
    return
