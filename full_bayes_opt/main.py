#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (YYYY-MM-DD)

"""
Bayesian optimization of black-box functions using two-part procedure.
  1. A fully Bayesian surrogate model to marginalize over alternative Gaussian Process hyperparameter choices, with the
     goal of making the procedure entirely hyper-parameter free.
  2. A trust-region method to improve the rate of convergence in the region of a putative minimum.
"""

from __future__ import division

import os
import numpy as np
import scipy.stats
import warnings


class BayesianOptimizer(object):
  def __init__(self, objective_function, bounding_box, acquisitor, stan_surrogate_model_path):
    """

    :param objective_function: function object - the function under minimization; must return a float
    :param bounding_box: iterable - contains the max and min for each dimension. Order of max and min is irrelevant,
     but all outputs depend on the order in which the _dimensions_ are supplied.
     For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
       [(2,-2),(-5,5)] or
       [[-2,2],[5,-5]] or
       np.array([[2,-2],[5,-5])
     or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
     Note that all computations are done on the scale of the supplied parameters. If you want to do something like fit
     a parameter that varies on the log scale, supply log-scaled coordinates and then do the appropriate
     back-transformation in the call to the objective_function.
    :param acquisitor: string - string must be in {"PI", "EQI", "EI", "UCB"}
      PI - probability of improvement
      EQI - expected quantile improvement
      EI - expected improvement
      UCB - upper confidence bound
    :param stan_surrogate_model_path: str - path to the model file
    """
    self._obj_fn = objective_function
    box = []
    for i, (x1, x2) in enumerate(bounding_box):
      if np.isclose(x1, x2):
        warnings.warn("The interval for dimension %d is too short to be plausible: [%s, %s]." %
                      (i, min(x1, x2), max(x1, x2)))
      box.append([min(x1, x2), max(x1, x2)])
    box = np.array(box)
    self._d = len(box)
    self.box = box

    self.acquisitor = acquisitor
    self._d = len(self.box)
    self.y = None
    self.x = None

    try:
      with open(stan_surrogate_model_path) as f:
        self._model_str = f.read()
    except FileNotFoundError:
      load_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), stan_surrogate_model_path)
      with open(load_path) as f:
        self._model_str = f.read()

  @property
  def d(self):
    return self._d

  @property
  def obj_fn(self):
    return self._obj_fn

  @property
  def model_str(self):
    return self._model_str

  def fit(self):
    return

  def sample_next(self, size=1):
    """
    samples a new point uniformly from within the bounding box
    :return:
    """
    out = np.zeros((0, self.d))
    for j in range(size):
      new_x = np.zeros(self.d)
      for i in range(self.d):
        new_x[i] = np.random.uniform(low=self.box[i, 0], high=self.box[i, 1], size=1)
      out = np.vstack((out, new_x))
    return np.matrix(out).reshape((size, self.d))
