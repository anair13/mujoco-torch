import matplotlib.pyplot as plt
import numpy as np

import json
from pprint import pprint
import rllab.viskit.core as core

read_tb = lambda: None
import glob
import os
import itertools

from contextlib import contextmanager
import sys, os

true_fn = lambda p: True
identity_fn = lambda x: x

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_exps(dirnames, filter_fn=true_fn, suppress_output=False):
    if suppress_output:
        with suppress_stdout():
            exps = core.load_exps_data(dirnames)
    else:
        exps = core.load_exps_data(dirnames)
    good_exps = []
    for e in exps:
        if filter_fn(e):
            good_exps.append(e)
    return good_exps

def read_params_from_output(filename, maxlines=200):
    if not filename in cached_params:
        f = open(filename, "r")
        params = {}
        for i in range(maxlines):
            l = f.readline()
            if not ":" in l:
                break
            kv = l[l.find("]")+1:]
            colon = kv.find(":")
            k, v = kv[:colon], kv[colon+1:]
            params[k.strip()] = v.strip()
        f.close()
        cached_params[filename] = params
    return cached_params[filename]

def prettify(p, key):
    """Postprocessing p[key] for printing"""
    return p[key]

def prettify_configuration(config):
    if not config:
        return ""
    s = ""
    for c in config:
        k, v = str(c[0]), str(c[1])
        x = ""
        x = k + "=" + v + ", "
        s += x
    return s[:-2]

def to_array(lists):
    """Converts lists of different lengths into a left-aligned 2D array"""
    M = len(lists)
    N = max(len(y) for y in lists)
    output = np.zeros((M, N))
    output[:] = np.nan
    for i in range(M):
        y = lists[i]
        n = len(y)
        output[i, :n] = y
    return output

def filter_by_flat_params(d):
    def f(l):
        for k in d:
            if l['flat_params'][k] != d[k]:
                return False
        return True
    return f

def comparison(exps, key, vary = ["expdir"], f=true_fn, smooth=identity_fn, figsize=(5, 3),
    xlabel="Number of env steps total", default_vary=False, xlim=None, ylim=None):
    """exps is result of core.load_exps_data
    key is (what we might think is) the effect variable
    vary is (what we might think is) the causal variable
    f is a filter function on the exp parameters"""
    plt.figure(figsize=figsize)
    plt.title("Vary " + " ".join(vary))
    plt.ylabel(key)
    y_data = {}
    x_data = {}
    def lookup(v):
        if v in l['flat_params']:
            return str(l['flat_params'][v])
        if v in default_vary:
            return str(default_vary[v])
        error
    for l in exps:
        if f(l) and l['progress']:
            label = " ".join([v + ":" + lookup(v) for v in vary])
            ys = y_data.setdefault(label, [])
            xs = x_data.setdefault(label, [])

            d = l['progress']
            x = d[xlabel]
            if key in d:
                y = d[key]

                y_smooth = smooth(y)
                x_smooth = x[:len(y_smooth)]
                ys.append(y_smooth)
                xs.append(x_smooth)
            else:
                print("not found", key)
                print(d.keys())
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    lines = []
    for label in sorted(y_data.keys()):
        ys = to_array(y_data[label])
        x = np.nanmean(to_array(x_data[label]), axis=0)
        y = np.nanmean(ys, axis=0)
        s = np.nanstd(ys, axis=0)
        plt.fill_between(x, y-s, y+s, alpha=0.2)
        line, = plt.plot(x, y, label=str(label))
        lines.append(line)
    plt.legend(handles=lines, bbox_to_anchor=(1.5, 0.75))

def split(exps,
    keys,
    vary = "expdir",
    split=[],
    f=true_fn,
    w="evaluator",
    smooth=identity_fn,
    figsize=(5, 3),
    suppress_output=False,
    xlabel="Number of env steps total",
    default_vary=False,
    xlim=None, ylim=None,):
    split_values = {}
    for s in split:
        split_values[s] = set()
    for l in exps:
        if f(l):
            for s in split:
                split_values[s].add(l['flat_params'][s])
    print(split_values)

    configurations = []
    for s in split_values:
        c = []
        for v in split_values[s]:
            c.append((s, v))
        configurations.append(c)
    for c in itertools.product(*configurations):
        fsplit = lambda exp: all([exp['flat_params'][k] == v for k, v in c]) and f(exp)
        # for exp in exps:
        #     print(fsplit(exp), exp['flat_params'])
        for key in keys:
            comparison(exps, key, vary, f=fsplit, smooth=smooth,
                figsize=figsize, xlabel=xlabel, default_vary=default_vary, xlim=xlim, ylim=ylim)
            plt.title(prettify_configuration(c) + " Vary " + " ".join(vary))

def ma_filter(N):
    return lambda x: moving_average(x, N)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

import itertools
def scatterplot_matrix(data1, data2, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = len(data1), len(data2)
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(16,16))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in itertools.product(range(numvars), range(numdata)):
        axes[i,j].scatter(data1[i], data2[j], **kwargs)
        label = "{:6.3f}".format(np.corrcoef([data1[i], data2[j]])[0, 1])
        axes[i,j].annotate(label, (0.1, 0.9), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    # for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
    #     axes[j,i].xaxis.set_visible(True)
    #     axes[i,j].yaxis.set_visible(True)

    return fig
