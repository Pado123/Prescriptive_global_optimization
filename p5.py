import pickle
import pandas as pd
import pm4py
import numpy as np
import random

import time
import tqdm
import hash_maps
import utils
from hash_maps import str_list, list_str
import next_act

# Preliminary imports of variables
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
    X_train.rename(columns={'REQUEST_ID': 'case:concept:name', 'ACTIVITY': 'concept:name', 'CE_UO': 'org:resource'},
                   inplace=True)
    df_rec.rename(columns={'REQUEST_ID': 'case:concept:name', 'ACTIVITY': 'concept:name', 'CE_UO': 'org:resource'},
                  inplace=True)
log = pm4py.convert_to_event_log(X_train)
# roles = pm4py.discover_organizational_roles(log)
# available_resources_list = utils.filter_resources_availability(set(X_train['org:resource'].unique()), p=.75)
activity_list = list(X_train['concept:name'].unique())
act_role_dict = pickle.load(open('act_role_dict.pkl', 'rb'))
cases_list = list(df_rec['case:concept:name'].unique())
traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name, thrs=0)
# traces_hash = pickle.load(open('traces_hash.pkl', 'rb'))
model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
quantitative_vars = pickle.load(open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'rb'))
qualitative_vars = pickle.load(open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'rb'))
df_sol = pd.read_csv('Results_mixed_r.csv', index_col=0)
df_sol['Case_id'] = df_sol['Case_id'].astype(str)
df_rec[case_id_name] = df_rec[case_id_name].astype(str)
delta_KPI = pickle.load(open('delta_kpi.pkl', 'rb'))
possible_permutations = dict()
initial_order = list(delta_KPI.keys())
np.random.seed(16184)

def generate_permutation(df_sol, idx_resource, delta_KPI):  # How to get a permutation from a solution, given the index

    # Copy the initial df
    df_sol2 = df_sol.copy()

    # Get the case id and the resource that'll be free
    case = df_sol2.iloc[idx_resource]
    case_activity = case['Activity_recommended']
    case_id = case['Case_id']
    res_to_reassign = case['Resource']

    # Get the trace and prepare trace for the prediction
    trace = df_rec[df_rec[case_id_name] == case_id].reset_index(drop=True)
    last = trace.loc[max(trace.index)].copy()
    resources_for_act = act_role_dict[case['Activity_recommended']]

    # put in all the columns which are not inferrable a null value
    for var in last.index:
        if var in (set(quantitative_vars).union(qualitative_vars)):
            last[var] = "none"

    partial_results = dict()
    for res in resources_for_act:
        last['org:resource'] = res
        # Create a vector with the actual prediction
        if pred_column == 'remaining_time':
            actual_prediction = model.predict(list(last[1:]))
        elif pred_column == 'independent_activity':
            actual_prediction = model.predict_proba(list(last[1:]))[0]  # activity case
        partial_results[res] = actual_prediction
    partial_results = dict(sorted(partial_results.items(), key=lambda item: item[1], reverse=False))
    best_new_res, new_expected_KPI = list(partial_results.keys())[1], list(partial_results.values())[0]
    second_new_res, third_new_res, = list(partial_results.keys())[2], list(partial_results.keys())[3]

    go = False
    i = 1  # Starts from one because zero's the best
    while not go:
        best_new_res = list(partial_results.keys())[i]
        try:
            index_to_remove = df_sol2[df_sol2['Resource'] == best_new_res].index[0]
            go = True
            print(f'preso alla {i} tentativo')
        except:
            i += 1
            print('ouch')

    df_sol2.at[idx_resource, 'Resource'], df_sol2.at[idx_resource, 'Expected KPI'] = best_new_res, new_expected_KPI

    # Update delta_KPI and df_sol with only the available cases
    df_sol2.drop(labels=index_to_remove, axis=0, inplace=True)
    df_sol2.reset_index(drop=True, inplace=True)
    updated_id = set(df_sol2['Case_id'])
    available_KPIs = {k: v for k, v in delta_KPI.items() if k not in updated_id}

    # Select the new couple activity-resource, in the KPI_first manner
    res_traces = dict()
    exit_cycles = False
    for trace_id in available_KPIs.keys():
        if exit_cycles == True:
            break
        for act in [el[1] for el in available_KPIs[trace_id]]:
            if res_to_reassign in act_role_dict[act]:
                chosen_act = act
                chosen_idx = trace_id
                exit_cycles = True
                break

    # Set baseline for predict the new KPI: Select the case and include the current act as traces' act
    pred_case = df_rec[df_rec[case_id_name] == chosen_idx]
    last = pred_case.loc[max(pred_case.index)].copy()
    last[activity_name] = chosen_act

    # put in all the columns which are not inferrable a null value
    for var in last.index:
        if var in (set(quantitative_vars).union(qualitative_vars)):
            last[var] = "none"
    last['org:resource'] = res_to_reassign

    if pred_column == 'remaining_time':
        reassigned_KPI = model.predict(list(last[1:]))
    elif pred_column == 'independent_activity':
        reassigned_KPI = model.predict_proba(list(last[1:]))[0]  # activity case

    upper_cases = initial_order[:initial_order.index(chosen_idx)][::-1]
    if upper_cases == list():
        line = pd.DataFrame(columns=df_sol2.columns)
        line.loc[0] = tuple((chosen_idx, chosen_act, res_to_reassign, reassigned_KPI))
        df_sol2 = pd.concat([line, df_sol2]).reset_index(drop=True)
    else:
        for case_ in upper_cases:
            if case_ not in df_sol2['Case_id'].values:
                continue
            else:
                index_of_case = df_sol2[df_sol2['Case_id'] == case_].index[0]
                line = pd.DataFrame(columns=df_sol2.columns)
                line.loc[0] = tuple((chosen_idx, chosen_act, res_to_reassign, reassigned_KPI))
                df_sol2 = pd.concat([df_sol2.iloc[:index_of_case], line, df_sol2.iloc[index_of_case:]]).reset_index(
                    drop=True)
                break

    print(f'The len is {len(df_sol2)}')
    return df_sol2


def generate_n_solutions_with_filtering_best_k(df_sol, n, k, delta_KPI):
    # n = number of generated solution
    # k = the best ones I keep
    solutions = dict()
    for l in range(n):
        index_to_replace = np.random.geometric(p=0.06, size=1)[0] #random.randint(0, 35)
        d = generate_permutation(df_sol=df_sol, idx_resource=index_to_replace, delta_KPI=delta_KPI)
        solutions[str(l)] = [d, np.sum(d['Expected KPI'])]
    solutions = {k: v for k, v in sorted(solutions.items(), key=lambda item: item[1][1])}
    solutions = {i: solutions[k] for k, i in zip(list(solutions.keys())[:k], range(k))}
    return solutions


def generate_solutions_tree(df_sol, height, length, generations_number, delta_KPI):
    solutions_tree = dict()
    solutions_tree['0'] = [df_sol, np.sum(df_sol['Expected KPI'])]
    h_ = 0
    if generations_number < length:
        raise ValueError('Solutions to filter are more than solutions to generate')

    while h_ <= height:
        print(f'Processing the level {h_} and the generated solutions are {len(solutions_tree)}')
        if len(solutions_tree.keys()) == 1 :
            # If it is empty, fill the first line
            partial_solutions = generate_n_solutions_with_filtering_best_k(df_sol, n=generations_number, k=length,
                                                                           delta_KPI=delta_KPI)
            partial_solutions = {str(h_) + '_' + str(i): partial_solutions[k] for k, i in
                                 zip(partial_solutions.keys(), range(len(partial_solutions.keys())))}
            for key in partial_solutions.keys():
                solutions_tree[key] = partial_solutions[key]

        else:
            # Read the datasets to analyze
            h_ += 1
            key_solutions_to_analyze = [key for key in solutions_tree.keys() if key.count('_') == h_]
            for df_key in key_solutions_to_analyze:
                df = solutions_tree[df_key][0]
                partial_solutions = generate_n_solutions_with_filtering_best_k(df, n=generations_number, k=length,
                                                                               delta_KPI=delta_KPI)
                partial_solutions = {str(df_key) + '_' + str(i): partial_solutions[k] for k, i in
                                     zip(partial_solutions.keys(), range(len(partial_solutions.keys())))}
                for key in partial_solutions.keys():
                    solutions_tree[key] = partial_solutions[key]
                with open('solutions_tree5.pkl', 'wb') as f:
                    pickle.dump(solutions_tree, f)
                    f.close()

    return solutions_tree

solutions_tree = generate_solutions_tree(df_sol, height=8, length=3, generations_number=5, delta_KPI=delta_KPI)
print('the solutions are ', len(solutions_tree))
solutions_tree = utils.filter_and_reorder_solutions_dict(solutions_tree)
print('and now ', len(solutions_tree))
pickle.dump(solutions_tree, open('solutions_tree5.pkl', 'wb'))

if __name__ == '__main__':
    print('Code executed')