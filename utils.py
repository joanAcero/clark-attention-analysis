from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import time


def _make_parent_dirs(path):
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)


def load_json(path):
  with open(path, 'r') as f:
    return json.load(f)


def write_json(o, path):
  _make_parent_dirs(path)
  with open(path, 'w') as f:
    json.dump(o, f)


def load_pickle(path):
  with open(path, 'rb') as f:
    try:
      return pickle.load(f)
    except UnicodeDecodeError:
      # pickles written by Python 2 (e.g. the data released with the paper)
      f.seek(0)
      return pickle.load(f, encoding='latin1')


def write_pickle(o, path):
  _make_parent_dirs(path)
  with open(path, 'wb') as f:
    pickle.dump(o, f, -1)


def logged_loop(iterable, n=None, **kwargs):
  if n is None:
    n = len(iterable)
  ll = LoopLogger(n, **kwargs)
  for i, elem in enumerate(iterable):
    ll.update(i + 1)
    yield elem


class LoopLogger(object):
  """Class for printing out progress/ETA for a loop."""

  def __init__(self, max_value=None, step_size=1, n_steps=25, print_time=True):
    self.max_value = max_value
    if n_steps is not None:
      self.step_size = max(1, max_value // n_steps)
    else:
      self.step_size = step_size
    self.print_time = print_time
    self.n = 0
    self.start_time = time.time()

  def step(self, values=None):
    self.update(self.n + 1, values)

  def update(self, i, values=None):
    self.n = i
    if self.n % self.step_size == 0 or self.n == self.max_value:
      if self.max_value is None:
        msg = 'On item ' + str(self.n)
      else:
        msg = '{:}/{:} = {:.1f}%'.format(self.n, self.max_value,
                                         100.0 * self.n / self.max_value)
        if self.print_time:
          time_elapsed = time.time() - self.start_time
          time_per_step = time_elapsed / self.n
          msg += ', ELAPSED: {:.1f}s'.format(time_elapsed)
          msg += ', ETA: {:.1f}s'.format((self.max_value - self.n)
                                         * time_per_step)
      if values is not None:
        for k, v in values:
          msg += ' - ' + str(k) + ': ' + ('{:.4f}'.format(v)
                                          if isinstance(v, float) else str(v))
      print(msg)
