#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-24 (year-month-day)

"""
"""

from __future__ import division


class BaseGaussianProcessOptimizer(object):
  def __init__(self):
    pass

  def gradient(self, z):
    pass

  def hessian(self, z):
    pass


class AnisotropicGaussianProcessOptimizer(BaseGaussianProcessOptimizer):
  def __init__(self):
    pass

  def kernel(self):
    pass

  def gradient(self, z):
    pass

  def hessian(self, z):
    pass


if __name__ == "__main__":
  pass
