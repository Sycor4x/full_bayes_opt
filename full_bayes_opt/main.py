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

import scipy.spatial as sp_spatial
import numpy as np
from .acquisition_fn import ExpectedImprovement, ExpectedQuantileImprovement

from .stan_helpers import stan_model_cache
import os


# TODO - code to prevent unnecessary model re-compilation
# TODO - include more optimization test functions
# TODO -

def prs_sample_size(q=0.05, p=0.95):
  """
  Computes the minimum sample size required to obtain function value in the smallest q quantile with probability p
  via pure random search. The most common usage assigns q = 0.05 and p = 0.95, whence one finds the rule-of-thumb of
  n = 60 random hyperparameter tuples.
  :param q: float in (0.0, 1.0) - the target quantile
  :param p: float in (0.0, 1.0) - the target probability
  :return: int
  """
  if not (0.0 < q < 1.0) or not (0.0 < p < 1.0):
    raise ValueError("Both p and q must be probabilities.")
  return np.log(1.0 - p) // np.log(q)


class BayesianOptimizer(object):
  def __init__(self, objective_function,
               bounding_box,
               acquisitor,
               stan_surrogate_model_path,
               objective_function_initial_values):
    """
    :param objective_function: function object - the function under minimization; must return a scalar real number
    :param objective_function_initial_values: dictionary of np arrays with keys "x" and "y".
      array "x" must have shape (N, d).
      array "y" must have N entries.
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

    # load and validate initial objective function values
    self.x = objective_function_initial_values["x"]
    self.y = objective_function_initial_values["y"]
    if self.x.shape[1] != self.d:
      raise ValueError("x has the wrong shape - x.shape[1] must equal len(bounding_box)=%d" % self.d)
    if self.y.size != len(self.x):
      raise ValueError("y must have len(x)=%d entries." % len(self.x))
    self.y.reshape((-1, 1))  # 2-dimensional array with 1 column

    with open(stan_surrogate_model_path) as f:
      self._stan_surrogate_model_code = f.read()

    pretty_name = os.path.basename(os.path.splitext(stan_surrogate_model_path)[0])
    self._surrogate = stan_model_cache(stan_code=self.stan_surrogate_model_code, model_name=pretty_name)

    self.surrogate_model_control_default = {
      "chains": 4,
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
      "n_jobs": -1
    }

  @property
  def d(self):
    return self._d

  @property
  def obj_fn(self):
    return self._obj_fn

  @property
  def stan_surrogate_model_code(self):
    return self._stan_surrogate_model_code

  @property
  def surrogate(self):
    return self._surrogate

  def explore(self, batch_size, surrogate_model_control):
    pars = ["y_tilde", "alpha", "rho", "sigma"]
    x_sim = self.sample_next(batch_size)
    data = {
      "N": self.y.size,
      "M": batch_size,
      "D": self.d,
      "y": self.y,
      "x": self.x,
      "x_tilde": x_sim,
    }
    fit = self.surrogate.sampling(data=data, pars=pars, **surrogate_model_control)
    new_x = self.acquisitor(y=self.y, x_sim=x_sim, stanfit_obj=fit)
    new_y = self.obj_fn(new_x)
    self.x = np.vstack((self.x, new_x))
    self.y = np.append(self.y, new_y)

  def fit(self, iter_opt=60, batch_size=100, surrogate_model_control=None):
    if not surrogate_model_control:
      surrogate_model_control = self.surrogate_model_control_default

    for i in range(iter_opt):
      self.explore(batch_size=batch_size, surrogate_model_control=surrogate_model_control)
    return self.get_best(), self.y.min()

  def get_best(self):
    best_ndx = self.y.argmin()
    return self.x[best_ndx, :]

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


class BayesianHybridOptimizer(BayesianOptimizer):
  def exploit(self):
    nearest_to_best = self.get_nearest_to_best()
    interp_data = np.vstack((np.ones(len(nearest_to_best)), np.square(nearest_to_best)))

    return

  def get_nearest_to_best(self):
    best_x = self.get_best()
    x_dist_to_best = sp_spatial.distance.cdist(best_x, self.x)
    nearest_to_best = np.argsort(x_dist_to_best)[:self.d + 2]
    return self.x[nearest_to_best, :]
