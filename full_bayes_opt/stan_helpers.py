#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-23 (year-month-day)

"""
cute caching function to save compiled stan models between sessions
"""

from __future__ import division

import pystan
import pickle
from hashlib import md5


def stan_model_cache(stan_code, model_name=None, **kwargs):
  """Use just as you would `stan`"""
  code_hash = md5(stan_code.encode("ascii")).hexdigest()
  if model_name is None:
    cache_fname = "stan_model_%s.pkl" % code_hash
  else:
    cache_fname = "stan_model_%s_%s.pkl" % (model_name, code_hash)
  try:
    with open(cache_fname, "rb") as f:
      sm = pickle.load(f)
  except:
    sm = pystan.StanModel(model_code=stan_code)
    with open(cache_fname, "wb") as f:
      pickle.dump(sm, f)
  else:
    print("Using cached StanModel")
  return sm


if __name__ == "__main__":
  model_code = """
      data {
        int<lower=0> N;
        int<lower=0,upper=1> y[N];
      }
      parameters {
        real<lower=0,upper=1> theta;
      }
      model {
        theta ~ beta(0.5, 0.5);
        for (n in 1:N)
          y[n] ~ bernoulli(theta);
      }
  """
  # with same model_code as before
  data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
  sm_model = stan_model_cache(stan_code=model_code)
  fit = sm_model.sampling(data=data)
  print(fit)

  new_data = dict(N=6, y=[0, 0, 0, 0, 0, 1])
  # the cached copy of the model will be used
  sm_model = stan_model_cache(stan_code=model_code)
  fit2 = sm_model.sampling(data=new_data)
  print(fit2)
