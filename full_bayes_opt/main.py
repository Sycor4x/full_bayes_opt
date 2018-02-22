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

import warnings

import numpy as np
import pystan

models = {"rbf": "gp_anisotropic_rbf.stan",
          "matern_2.5": None}


def prs_sample_size(q, p):
  """
  Computes the minimum sample size required to obtain function value in the smallest q quantile with probability p
  via pure random search. The most common usage assigns q = 0.05 and p = 0.95, whence one finds the rule-of-thumb of
  n = 60 random hyperparameter tuples.
  :param q: float in (0.0, 1.0) - the target quantile (higher = better)
  :param p: float in (0.0, 1.0) - the target probability (higher = better)
  :return: int
  """
  if not (0.0 < q < 1.0) or not (0.0 < p < 1.0):
    raise ValueError("Both p and q must be probabilities.")
  return np.log(1.0 - p) // np.log(q)


class BayesianOptimizer(object):
  def __init__(self, objective_function,
               bounding_box,
               acquisitor,
               objective_function_initialization=("random", 8),
               stan_surrogate_model_path="",
               ):
    """

    :param objective_function: function object - the function under minimization; must return a float
    :param objective_function_initialization:
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

    initialization_method, initialization_data = objective_function_initialization
    if not isinstance(initialization_method, str):
      raise ValueError("`initialization_method` must be str but instead supplied %s." % type(initialization_method))

    if initialization_method == "precomputed":
      self.x, self.y, self.y_var = initialization_data
    elif initialization_method == "random":
      self.x = self.sample_next(initialization_data)
      for next_x in self.x:
        next_y, next_y_var = self.obj_fn(next_x)
        self.y = np.vstack((self.y, next_y))
        self.y_var = np.vstack((self.y_var, next_y_var))
    else:
      raise ValueError("`initialization_method` is %s, which is not recognized." % initialization_method)

    self.y = None
    self.y_var = None
    self.x = None

    with open(stan_surrogate_model_path) as f:
      self._model_str = f.read()

    self._surrogate = pystan.StanModel(file=self.model_str)

    self.surrogate_model_control_default = {"chains": 4,
                                            "iter": 1000,
                                            "warmup": None,
                                            "thin": 1,
                                            "seed": None,
                                            "init": "random",
                                            "sample_file": None,
                                            "diagnostic_file": None,
                                            "verbose": False,
                                            "algorithm": "NUTS",
                                            "control": None,
                                            "n_jobs": -1}

  @property
  def d(self):
    return self._d

  @property
  def obj_fn(self):
    return self._obj_fn

  @property
  def model_str(self):
    return self._model_str

  @property
  def surrogate(self):
    return self._surrogate

  def fit(self, iter_opt=60, batch_size=50, surrogate_model_control=None):
    parlist = ["y_tilde", "alpha", "rho", "sigma", "eta"]
    if not surrogate_model_control:
      surrogate_model_control = self.surrogate_model_control_default

    for i in range(iter_opt):
      x_tilde = self.sample_next(batch_size)
      self.surrogate.sampling(pars=parlist, **surrogate_model_control)
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
