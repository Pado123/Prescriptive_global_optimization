import pickle
import pandas as pd
import pm4py
import numpy as np
import random
import os
import glob

import time
import tqdm
import hash_maps
import utils
from hash_maps import str_list, list_str
import next_act
random.seed(1618)



experiment_name = 'experiment_files'
case_id_name = 'REQUEST_ID'
X_train, X_test, y_train, y_test = utils.import_vars(experiment_name=experiment_name, case_id_name=case_id_name)

case_id_name = 'case:concept:name'
pred_column = 'remaining_time'
if 'ACTIVITY' in X_train.columns:
    X_train.rename(columns={'REQUEST_ID': 'case:concept:name', 'ACTIVITY': 'concept:name', 'CE_UO': 'org:resource'}, inplace=True)
log = pm4py.convert_to_event_log(X_train)
act_role_dict = pickle.load(open('act_role_dict.pkl', 'rb'))

delta_kpi = pickle.load(open('delta_kpi.pkl', 'rb'))
# evaluation_dict = pickle.load(open('evaluation_dict.pkl', 'rb'))
solutions_tree = pickle.load(open('solutions_tree.pkl', 'rb'))
# resources_cases_dict = generate_case_resource_dict(solutions_tree)
resources_cases_dict = pickle.load(open('resources_cases_dict.pkl', 'rb'))

# act_role_dict_eval = generate_evaluation_dict(X_train, y_train, act_role_dict)
act_role_dict_eval = pickle.load(open('act_role_dict_eval.pkl', 'rb'))

def evaluate_resource_case_variability(solutions_tree):
    for res in solutions_tree[list(solutions_tree.keys())[0]][0]['Resource']:
        a = set()
        for key in solutions_tree.keys():
            try :
                df = solutions_tree[key][0]
                a.add(df[df['Resource'] == res]['Case_id'].values[0])
            except : None
        print(f'for the res {res} there are {len(a)} cases')

def generate_evaluation_dict(X_train, y_train, act_role_dict, pred_column='remaining_time'):

    log = pm4py.convert_to_event_log(X_train)
    X_train['y'] = y_train.values
    df_for_evaluation = X_train.copy()
    act_role_dict_eval = dict()

    for act in tqdm.tqdm(act_role_dict.keys()):
        act_role_dict_eval[act] = dict()

    for act in tqdm.tqdm(act_role_dict.keys()):
        for res in act_role_dict[act]:
            score = df_for_evaluation.loc[(df_for_evaluation['concept:name']==act) & (df_for_evaluation['org:resource'] == res), ['y']].mean().values[0]

            if act in act_role_dict_eval.keys():
                act_role_dict_eval[act][res] = score

    return act_role_dict_eval

def generate_choice_customized_prob(weights=[.45, .165, .115, .06, 0.045, .065, .025, .03, .015, .03, .02, .02]):
    return random.choices(range(0,12), weights)

def generate_case_resource_dict(solutions_tree): #6 MIN

    c =0
    resources_cases_dict = dict()
    res_list = set(solutions_tree[list(solutions_tree.keys())[0]][0]['Resource'].unique())
    for key in solutions_tree.keys():
        res_list = res_list.intersection(set(solutions_tree[key][0]['Resource'].unique()))
    res_list = list(res_list)
    for res in tqdm.tqdm(res_list):
        resources_cases_dict[res] = dict()
        if res != 'missing':
            for key in solutions_tree.keys():

                try :
                    case, score = \
                    solutions_tree[key][0][solutions_tree[key][0]['Resource'] == res][['Case_id', 'Expected KPI']].values[0]
                    if case in resources_cases_dict[res].keys():
                        resources_cases_dict[res][case].append(score)
                    if case not in resources_cases_dict[res].keys():
                        resources_cases_dict[res][case] = [score]
                except :
                    c+=1
                    print('evabb', c)

    #Replace it with its associated KPI
    for res in tqdm.tqdm(resources_cases_dict.keys()):
        for case in resources_cases_dict[res]:
            resources_cases_dict[res][case] = np.mean(resources_cases_dict[res][case])
    for res in tqdm.tqdm(resources_cases_dict.keys()):
        resources_cases_dict[res] = {k: v for k, v in sorted(resources_cases_dict[res].items(), key=lambda item: item[1])}

    return resources_cases_dict

def evaluate_set(solutions_tree, act_role_dict_eval, customized=True): #TODO: this function has to be completed

    # Get the resources of the tree and rename the keys for convenience
    res_list = set(solutions_tree[list(solutions_tree.keys())[0]][0]['Resource'].unique())
    for key in solutions_tree.keys():
        res_list = res_list.intersection(set(solutions_tree[key][0]['Resource'].unique()))
    res_list = list(res_list)
    new_keys = range(len(solutions_tree.keys()))
    solutions_tree = dict(zip(new_keys, [i[0] for i in solutions_tree.values()]))

    #Random shuffle the resources for their arrival order (it is done in-place)
    random.shuffle(res_list)

    # Generate an id-resource dictionary for the pointwise situation
    cases_resources_dict = generate_case_resource_dict(solutions_tree)



    #Following their arrival order, evaluate the choice of the list
    cumulative_avg_kpi = 0
    idx_res = 0
    chosen_case = None
    cases_done = set([None])

    if customized == True:

        for arrived_res in tqdm.tqdm(res_list):
            print(f'the resource is ', res_list.index(arrived_res))
            while chosen_case in cases_done:

                #Generate its choiche between the given probability (paper referenced)
                choice_idx = generate_choice_customized_prob()[0]

                #Get the best-10 available cases
                cases_available_for_res_with_kpi = dict()
                for case in resources_cases_dict[arrived_res].keys():
                    if case not in cases_done:
                        cases_available_for_res_with_kpi[case] = resources_cases_dict[arrived_res][case]
                    if len(cases_available_for_res_with_kpi.keys()) == choice_idx:
                        break




            print(f'the current score for case {chosen_case} and res {arrived_res} is {act_role_dict_eval[act_sol][arrived_res]}')
            if choice_idx == 0:
                idx_res +=1

        return cumulative_avg_kpi


def merge_sol(sol1, sol2):

    #Rename the keys of the dictionary for having just an order
    r1, r2 = range(len(sol1)), range(len(sol1), len(sol1) + len(sol2))
    sol1 = {i: sol1[k] for i,k in zip(r1, sol1.keys())}
    sol2 = {i: sol2[k] for i,k in zip(r2, sol2.keys())}

    #Merge and filter dictionaries
    sol1 = sol1 | sol2
    print(f'The len of dictionaries was {len(sol1)} and {len(sol2)}')
    sol1 = utils.filter_and_reorder_solutions_dict(sol1)
    print(f'After the merging procedure is {len(sol1)}, {r2[-1] - len(sol1)} solutions have been removed')
    return sol1


def cut_dict(diz, n=200):
    return {k:diz[k] for k in list(diz.keys())}

def merge_sol_all():

    #Set the directory in which there are the solutions
    if os.getcwd()[-5:]!='trees':
        os.chdir('trees')

    names = glob.glob('*')
    final_sol = dict()
    s1, s2 = pickle.load(open(names[0], 'rb')), pickle.load(open(names[1], 'rb'))
    final_sol = merge_sol(s1, s2)

    for name in glob.glob('*'):
        print(name)
        try:
            partial_solution = pickle.load(open(name, 'rb'))
        except :
            print(f'db {name}')
        final_sol = merge_sol(final_sol, partial_solution)

    if os.getcwd()[-5:]=='trees':
        os.chdir('..')

    return final_sol

a = merge_sol_all()
pickle.dump(a, open('glob_sol_2.pkl', 'wb'))
# a = evaluate_set(solutions_tree=solutions_tree, act_role_dict_eval=act_role_dict_eval, customized=True)
# b = evaluate_set(solutions_tree=solutions_tree, act_role_dict_eval=act_role_dict_eval, customized=False)

if __name__ == '__main__':
    b = 'not implemented'
    print(f'We have generated {len(a)} different solutions')
    print(f'Code executed the cumulative avg time is {a} against the time without our ranking is {b}')



