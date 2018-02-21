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
import numpy as np
import scipy.stats
import warnings


class BayesianOptimizer(object):
  def __init__(self, objective_function, bounding_box, acquisition_function, stan_surrogate_model_path):
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
    :param acquisition_function: string - string must be in {"PI", "EQI", "EI", "UCB"}
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

    self.acquisition_function = acquisition_function
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

  def EQI(self, x_tilde, y_sim, tau):
    tau_sq = np.power(tau, 2.0)
    y_sim_bar = y_sim.mean(axis=0)
    y_sim_var = y_sim.std(ddof=1, axis=0)
    min_q = self.y.mean(axis=0) - eqi_coef * self.y.std(ddof=1, axis=0)
    s_sq_Q = y_sim_var ** 2
    s_sq_Q /= y_sim_var + tau_sq
    giant_blob = tau_sq * y_sim_var
    giant_blob /= tau_sq * y_sim_var
    m_Q = y_sim_bar + eqi_coef * np.sqrt(giant_blob)
    s_Q = np.sqrt(s_sq_Q)
    eqi = (min_q - m_Q) * scipy.stats.norm(min_q, loc=m_Q, scale=s_Q)
    eqi += s_Q * scipy.stats.norm(min_q, loc=m_Q, scale=s_Q)
    best_ndx = eqi.argmax()
    return x_tilde[best_ndx, :]

  def EI(self, x_sim, y_sim):
    y_tilde_bar = y_sim.mean(axis=0)
    y_tilde_s = y_sim.std(ddof=1, axis=0)
    best_y = self.y.min()
    ei = (best_y - y_tilde_bar) * scipy.stats.norm.cdf(best_y, loc=y_tilde_bar, scale=y_tilde_s)
    ei += y_tilde_s * scipy.stats.norm.ppf(best_y, loc=y_tilde_bar, scale=y_tilde_s)
    best_ndx = ei.argmax()
    return x_sim[best_ndx, :]

  def LCB(self, x_sim, y_sim):
    y_tilde_bar = y_sim.mean(axis=0)
    y_tilde_s = y_sim.std(ddof=1, axis=0)
    lcb = y_tilde_bar - lcb_coef * y_tilde_s
    best_ndx = lcb.argmin()
    return x_sim[best_ndx, :]

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
