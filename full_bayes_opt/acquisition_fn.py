#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-21 (year-month-day)

"""
Acquisition functions for Bayesian optimization.
"""

from __future__ import division

import numpy as np
import scipy.stats


class AcquisitionFunction(object):
  def unpack(self, stanfit_obj):
    param_dict = stanfit_obj.extract(["y_tilde", "sigma"])
    y_tilde = param_dict["y_tilde"]
    tau = param_dict["sigma"]
    return y_tilde, tau

  def prep(self, y):
    y_bar = y.mean(axis=0)
    y_sd = y.std(ddof=1.0, axis=0)
    return y_bar, y_sd


class ExpectedImprovement(AcquisitionFunction):
  def __call__(self, y, x_sim, stanfit_obj, *args, **kwargs):
    y_sim, _ = self.unpack(stanfit_obj)
    y_sim_bar, y_sim_var = self.prep(y_sim)

    best_y = y.min()
    ei = (best_y - y_sim_bar) * scipy.stats.norm.cdf(best_y, loc=y_sim_bar, scale=y_sim_var)
    ei += y_sim_var * scipy.stats.norm.ppf(best_y, loc=y_sim_bar, scale=y_sim_var)
    best_ndx = ei.argmax()
    return x_sim[best_ndx, :]


class ExpectedQuantileImprovement(AcquisitionFunction):
  def __init__(self, beta):
    # validate inputs
    if not isinstance(beta, float):
      raise ValueError("argument beta must be float but supplied %s" % beta)
    if not (0.5 <= beta < 1.0):
      raise ValueError("argument beta must be in [0.5, 1.0) but supplied %s." % beta)
    # fix this constant
    self._eqi_coef = scipy.stats.norm(beta)

  @property
  def eqi_coef(self):
    return self._eqi_coef

  def __call__(self, y, x_sim, stanfit_obj, *args, **kwargs):
    y_sim, tau = self.unpack(stanfit_obj)
    tau_sq = np.power(tau, 2.0)
    y_sim_bar, y_sim_var = self.prep(y_sim)

    s_sq_Q = y_sim_var ** 2
    s_sq_Q /= y_sim_var + tau_sq
    giant_blob = tau_sq * y_sim_var
    giant_blob /= tau_sq + y_sim_var

    m_Q = y_sim_bar + self.eqi_coef * np.sqrt(giant_blob)  # yes, that's supposed to be a plus sign
    s_Q = np.sqrt(s_sq_Q)

    min_q = y.mean(axis=0) + self.eqi_coef * y.std(ddof=1, axis=0)

    eqi = (min_q - m_Q) * scipy.stats.norm(min_q, loc=m_Q, scale=s_Q)
    eqi += s_Q * scipy.stats.norm(min_q, loc=m_Q, scale=s_Q)

    best_ndx = eqi.argmax()
    return x_sim[best_ndx, :]


class LowerConfidenceBound(AcquisitionFunction):
  def __call__(self, y, x_sim, stanfit_obj, *args, **kwargs):
    y_sim, tau = self.unpack(stanfit_obj)
    y_sim_bar, y_sim_var = self.prep(y_sim)

    lcb = y_sim_bar - lcb_coef * y_sim_var
    best_ndx = lcb.argmin()
    return x_sim[best_ndx, :]
