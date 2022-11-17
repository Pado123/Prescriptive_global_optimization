import argparse
import json
import os
import shutil
import warnings
import pickle
import time
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
print(f'THE INITIAL TIME IS {time.time()}')

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
        import ipdb; ipdb.set_trace()
        next_activities.sort_values(by='kpi_rel', inplace=True)
        delta_KPI[str(case_id)] = list()
        for line in next_activities.index:
            delta_KPI[str(case_id)].append((actual_prediction - next_activities.iloc[line]['kpi_rel'], next_activities.iloc[line]['Next_act']))
        # delta_KPI[str(case_id)] = [(actual_prediction - next_activities['kpi_rel'][i], next_activities['Next_act'][i]) for i in range(len(next_activities))]
    except:
        c+=1

print(f'The number of missed cases is {c}, and the final time {time.time()}') #10% of missed cases, just 234 different prediction values, maybe the predictor is too much lowered
delta_KPI = dict(sorted(delta_KPI.items(), key=lambda item: item[1][0], reverse=True))
# delta_KPI = pickle.load(open('delta_kpi.pkl', 'rb'))
Sol = list()
df_rec['case:concept:name'] = [str(i) for i in df_rec['case:concept:name']]

c=False
print(f'THE INITIAL TIME IS {time.time()}')
for trace_idx in tqdm.tqdm(list(delta_KPI.keys())):
    if available_resources_list == set():
        print('Resources finished')
        break
    delta_KPIa = np.array(delta_KPI[trace_idx])
    delta_KPIa = delta_KPIa[delta_KPIa[:,1].argsort()[::-1]]
    for act in delta_KPIa[:,1]:
        try:
            resources_for_act = act_role_dict[act]
        except:
            resources_for_act = None
        if set(resources_for_act).intersection(available_resources_list) != set():
            break
        else:
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
        print(f'b2')
        c=True
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
        best_res = dict(sorted(partial_results.items(), key=lambda item: item[1], reverse=False))
        best_res, expected_KPI = list(best_res.keys())[0], list(best_res.values())[0]
        available_resources_list.remove(best_res)
    except:
        print('bah')

    Sol.append((trace_idx, act, best_res, expected_KPI))

pickle.dump(Sol, open('Sol.pkl', 'wb'))
pickle.dump(delta_KPI, open('Delta_KPI.pkl', 'wb'))
df_sol = pd.DataFrame(Sol, columns=['Case_id', 'Activity_recommended', 'Resource', 'Expected KPI'])
df_sol.to_csv('Results_mixed_r.csv')
print(f'THE FINAL TIME IS {time.time()}')



if __name__ == '__main__':
    print('ihihih')