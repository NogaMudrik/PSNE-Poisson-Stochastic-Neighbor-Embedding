# -*- coding: utf-8 -*-
"""
DIMPO utility functions: checkEmptyList, save_fig, assertion helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def checkEmptyList(obj):
    """Check if obj is an empty list."""
    return isinstance(obj, list) and len(obj) == 0

def add_labels(ax, xlabel='X', ylabel='Y', zlabel='', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], ylabel_params = {},zlabel_params = {}, xlabel_params = {},  title_params = {}):
  """
    Add labels, titles, limits, etc. to a figure.

    Parameters:
    ax (subplot): The subplot to be edited.
    xlabel (str, optional): The label for the x-axis. Defaults to 'X'.
    ylabel (str, optional): The label for the y-axis. Defaults to 'Y'.
    zlabel (str, optional): The label for the z-axis. Defaults to ''.
    title (str, optional): The title for the plot. Defaults to ''.
    xlim (list or tuple, optional): The limits for the x-axis. Defaults to None.
    ylim (list or tuple, optional): The limits for the y-axis. Defaults to None.
    zlim (list or tuple, optional): The limits for the z-axis. Defaults to None.
    xticklabels (array, optional): The labels for the x-axis tick marks. Defaults to np.array([None]).
    yticklabels (array, optional): The labels for the y-axis tick marks. Defaults to np.array([None]).
    xticks (list, optional): The positions for the x-axis tick marks. Defaults to [].
    yticks (list, optional): The positions for the y-axis tick marks. Defaults to [].
    legend (list, optional): The legend for the plot. Defaults to [].
    ylabel_params (dict, optional): Additional parameters for the y-axis label. Defaults to {}.
    zlabel_params (dict, optional): Additional parameters for the z-axis label. Defaults to {}.
    xlabel_params (dict, optional): Additional parameters for the x-axis label. Defaults to {}.
    title_params (dict, optional): Additional parameters for the title. Defaults to {}.

  """

  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**ylabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)

  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend)       > 0:  ax.legend(legend)
  
  
def remove_edges(ax, include_ticks = True, top = False, right = False, bottom = True, left = True):
    """
    Remove the specified edges (spines) of the plot and optionally the ticks of the plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object of the plot.
    include_ticks : bool, optional
        Whether to include the ticks, by default False.
    top : bool, optional
        Whether to remove the top edge, by default False.
    right : bool, optional
        Whether to remove the right edge, by default False.
    bottom : bool, optional
        Whether to remove the bottom edge, by default False.
    left : bool, optional
        Whether to remove the left edge, by default False.
    
    Returns
    -------
    None
    """    
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    

def str2bool(str_to_change):
    """
    Transform a string representation of a boolean value to a boolean variable.
    
    Parameters:
    str_to_change (str): String representation of a boolean value
    
    Returns:
    bool: Boolean representation of the input string
    
    Example:
        str2bool('true') -> True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')  or (str_to_change.lower()  == 't') or (str_to_change.lower() == 'y')
    return str_to_change

    
def create_legend(dict_legend, size = 30, save_formats = ['.png','.svg'], 
                  save_addi = 'legend' , dict_legend_marker = {}, 
                  marker = '.', style = 'plot', s = 500, to_save = True, plot_params = {'lw':10}, to_sort_keys = False,
                  save_path = os.getcwd(), params_leg = {}, fig = [], ax = [], figsize = None, to_remove_edges =  True,
                  transparent = True,
                  dict_legend_keys = [], dict_legend_ls = {}):
    
    if not os.path.exists(save_path):
        print('path %s does not exist!'%save_path)
        to_create_the_path = str2bool(input('do you want to create the path \n %s'%save_path))
        if to_create_the_path:
            os.makedirs(save_path)
        
        
        
    if 'ls' in plot_params: 
        ls = plot_params['ls']
        del plot_params['ls']
    else:
        ls = '-'
    if len(dict_legend_keys) == 0:
        dict_legend_keys = list(dict_legend.keys())
        if to_sort_keys: # and dict_legend_keys:
            dict_legend_keys = np.sort(dict_legend_keys)
        
    assert np.array([el in dict_legend for el in dict_legend_keys]).all(), 'pay attention! some elemenets you provided in dict_legend_keys do not exist in dict_legend!'
    if set(dict_legend_keys) != set(list(dict_legend.keys())): print('pay attention! not all keys of dict_legend exist in dict_legend_keys!')
    
    
    if not isinstance(figsize,tuple) and not figsize:
        width = np.max([len(str(el)) for el in dict_legend_keys])
        length = len( dict_legend_keys)
        figsize = (3+(width*size/100)*params_leg.get('ncol', 1) ,3+(length*size/60)/params_leg.get('ncol', 1))
        
    if checkEmptyList(fig) and checkEmptyList(ax):
        fig, ax = plt.subplots(figsize = figsize)    
    else:
        to_remove_edges = False
    
        
    if checkEmptyList(fig) != checkEmptyList(ax):    
        raise ValueError('??')
    
    if style == 'plot':
        [ax.plot([],[], 
                 c = dict_legend[area], label = area, marker = dict_legend_marker.get(area),
                 ls = dict_legend_ls.get(area, ls), **plot_params) for area in dict_legend_keys]
    else:
        if len(dict_legend_marker) == 0:
            [ax.scatter([],[], s=s,c = dict_legend.get(area), label = area, marker = marker, **plot_params) for area in dict_legend_keys]
        else:
            [ax.scatter([],[], s=s,c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend_keys]
    ax.legend(prop = {'size':size},**params_leg)
    
    if to_remove_edges :
        remove_edges(ax, left = False, bottom = False, include_ticks = False)
    fig.tight_layout()
    
    if to_save:
        
        [fig.savefig(save_path + os.sep + 'legend_areas_%s%s'%(save_addi,type_save), transparent=transparent) 
         for type_save in save_formats]
        print('legend saved in %s'%(save_path + os.sep + 'legend_areas_%s.png'%(save_addi)))
        
        
        
        
        
        
def save_fig(name_fig, fig, save_path = '', formats = ['png','svg'], save_params = {}, verbose = True) :
    if len(save_path) == 0:
        save_path = os.getcwd()
    if 'transparent' not in save_params:
        save_params['transparent'] = True
    [fig.savefig(save_path + os.sep + '%s.%s'%(name_fig, format_i), **save_params) for format_i in formats]
    if verbose:
        print('saved figure: %s'%(save_path + os.sep + '%s.%s'%(name_fig, 'png')))


def assert_finite(arr, name):
    """Assert array has no NaN or Inf."""
    assert isinstance(arr, np.ndarray), '%s must be a numpy array, got %s' % (name, type(arr))
    assert np.all(np.isfinite(arr)), '%s contains NaN or Inf. NaN count: %d, Inf count: %d' % (
        name, np.sum(np.isnan(arr)), np.sum(np.isinf(arr)))


def assert_shape(arr, expected_shape, name):
    """Assert array has expected shape."""
    assert arr.shape == expected_shape, '%s has shape %s, expected %s' % (name, str(arr.shape), str(expected_shape))


def assert_nonnegative(arr, name):
    """Assert all elements >= 0."""
    assert np.all(arr >= 0), '%s contains negative values. Min value: %f' % (name, np.min(arr))


def assert_probability_matrix(arr, name, tol=1e-6):
    """Assert arr sums to 1 and is non-negative."""
    assert_nonnegative(arr, name)
    total = np.sum(arr)
    assert np.abs(total - 1.0) < tol, '%s does not sum to 1. Sum: %.10f, diff: %.2e' % (name, total, np.abs(total - 1.0))
