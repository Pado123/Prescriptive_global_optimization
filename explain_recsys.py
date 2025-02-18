#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:29:54 2022

@author: padela
"""

import catboost
from catboost import *
import shap
shap.initjs()

import urllib3
urllib3.disable_warnings()
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def evaluate_shap_vals(trace, model, X_test, case_id_name):
    trace = trace.iloc[-1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(trace.iloc[:1])
    return shap_values[0]
   

def plot_explanations_recs(groundtruth_explanation, explanations, idxs_chosen, last, experiment_name, trace_idx, act):

    # Python dictionary
    expl_df = {"Following Recommendation": [i for i in explanations[idxs_chosen].sort_values(ascending=False).values],
                          "Actual Value": [i for i in groundtruth_explanation[idxs_chosen].sort_values(ascending=False).values]};

    last = last[idxs_chosen]

    feature_names = [str(i) for i in last.index]
    feature_values = [str(i) for i in last.values]

    index = [feature_names[i]+'='+feature_values[i] for i in range(len(feature_values))]
    # Python dictionary into a pandas DataFrame

    dataFrame = pd.DataFrame(data=expl_df)

    dataFrame.index = index

    dataFrame.plot.barh(rot=0,
                        title=f"How variables contribution on \n KPI changes, ",
                        color=['darkgreen', 'darkred'])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig(f'figure.png')
    plt.close()