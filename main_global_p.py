import argparse
import json
import os
import shutil
import warnings
import pickle

# REFACTOR
from os.path import join

import numpy as np
import pandas as pd
import tqdm

# Converter
import pm4py

import hash_maps
import next_act
import utils
from IO import read, folders, create_folders
from load_dataset import prepare_dataset

experiment_name = 'experiment_files'
case_id_name = 'REQUEST_ID'

X_train, X_test, y_train, y_test = utils.import_vars(experiment_name=experiment_name, case_id_name=case_id_name)
activity_name = 'concept:name'
# df_rec = utils.get_test(X_test, case_id_name).reset_index(drop=True)
df_rec = pickle.load(open('df_rec.pkl', 'rb'))

columns = X_test.columns
case_id_name = 'case:concept:name'
pred_column = 'remaining_time'

if 'ACTIVITY' in X_train.columns:
    X_train.rename(columns={'REQUEST_ID': 'case:concept:name', 'ACTIVITY': 'concept:name', 'CE_UO': 'org:resource'}, inplace=True)
    df_rec.rename(columns={'REQUEST_ID': 'case:concept:name', 'ACTIVITY': 'concept:name', 'CE_UO': 'org:resource'},
                   inplace=True)
log = pm4py.convert_to_event_log(X_train)
roles = pm4py.discover_organizational_roles(log)
available_resources_list = set(X_train['org:resource'].unique())
activity_list = list(X_train['concept:name'].unique())
act_role_dict = dict()
cases_list = list(df_rec['case:concept:name'].unique())
for act in activity_list:
    for idx in range(len(roles)):
        if act in roles[idx][0]:
            act_role_dict[act] = list(roles[idx][1].keys())



traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name,
                                     thrs=0)
# print('Hash-map created')
# print('Analyze variables...')
# quantitative_vars, qualitative_trace_vars, qualitative_vars = utils.variable_type_analysis(X_train, case_id_name,
#                                                                                            activity_name)
# warnings.filterwarnings("ignore")
# print('Variable analysis done')
# pickle.dump(quantitative_vars, open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'wb'))
# pickle.dump(qualitative_vars, open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'wb'))
# pickle.dump(traces_hash, open(f'explanations/{experiment_name}/traces_hash.pkl', 'wb'))


model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)

# traces_hash = pickle.load(open('gui_backup/transition_system.pkl', 'rb'))
quantitative_vars = pickle.load(open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'rb'))
qualitative_vars = pickle.load(open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'rb'))

#Associate prediction to KPI
delta_KPI = dict()
c=0
for case_id in tqdm.tqdm(cases_list):
    trace = df_rec[df_rec[case_id_name] == case_id].reset_index(drop=True)
    # trace = trace.iloc[:(randrange(len(trace)) - 1)]
    trace = trace.reset_index(drop=True)

    # take activity list
    try:
        acts = list(df_rec[df_rec[case_id_name] == case_id].reset_index(drop=True)[activity_name])

        # Remove the last (it has been added because of the evaluation)
        trace = trace.iloc[:-1].reset_index(drop=True)

        next_activities, actual_prediction = next_act.next_act_kpis(trace, traces_hash, model, pred_column, case_id_name,
                                                                    activity_name,
                                                                    quantitative_vars, qualitative_vars, encoding='aggr-hist')

        delta_KPI[str(case_id)] = [actual_prediction - np.min(next_activities['kpi_rel']),
                                   next_activities[next_activities['kpi_rel']==np.min(next_activities['kpi_rel'])]['Next_act'].values[0]]
    except:
        c+=1

print(f'The number of missed cases is {c}') #10% of missed cases, just 234 different prediction values, maybe the predictor is too much lowered

delta_KPI = dict(sorted(delta_KPI.items(), key=lambda item: item[1][0], reverse=True))
Sol = list()
df_rec['case:concept:name'] = [str(i) for i in df_rec['case:concept:name']]

for trace_idx in tqdm.tqdm(delta_KPI.keys()):
    if available_resources_list == set():
        print('rotto')
        break
    best_activity = delta_KPI[trace_idx][1]
    try:
        resources_for_act = act_role_dict[best_activity]
    except:
        resources_for_act = None
        print(f'for trace {trace_idx} there\'s no resource available')
        continue


    pred_case = df_rec[df_rec[case_id_name]==trace_idx]
    last = pred_case.loc[max(pred_case.index)].copy()
    last_act = last[activity_name]

    # put in all the columns which are not inferrable a null value
    for var in last.index:
        if var in (set(quantitative_vars).union(qualitative_vars)):
            last[var] = "none"

    partial_results = dict()
    resources_for_act = set(resources_for_act).intersection(available_resources_list)
    if resources_for_act == set():
        print(f'there is no resource available for case {trace_idx}')
        continue

    for res in resources_for_act:
        last['org:resource'] = res
        # Create a vector with the actual prediction
        if pred_column == 'remaining_time':
            actual_prediction = model.predict(list(last[1:]))
        elif pred_column == 'independent_activity':
            actual_prediction = model.predict_proba(list(last[1:]))[0]  # activity case
        partial_results[res] = actual_prediction

    try:
        best_res = dict(sorted(partial_results.items(), key=lambda item: item[1], reverse=True))
        best_res = list(best_res.keys())[0]
        available_resources_list.remove(best_res)
    except:
        print('bah')

    Sol.append((trace_idx, best_activity, best_res))






if __name__ == '__main__':
    print('ihihih')