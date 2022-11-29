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
random.seed(1618)


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

experiment_name = 'experiment_files'
case_id_name = 'REQUEST_ID'
X_train, X_test, y_train, y_test = utils.import_vars(experiment_name=experiment_name, case_id_name=case_id_name)

case_id_name = 'case:concept:name'
pred_column = 'remaining_time'
if 'ACTIVITY' in X_train.columns:
    X_train.rename(columns={'REQUEST_ID': 'case:concept:name', 'ACTIVITY': 'concept:name', 'CE_UO': 'org:resource'}, inplace=True)
log = pm4py.convert_to_event_log(X_train)
act_role_dict = pickle.load(open('act_role_dict.pkl', 'rb'))
# act_role_dict_eval = generate_evaluation_dict(X_train, y_train, act_role_dict)
act_role_dict_eval = pickle.load( open('act_role_dict_eval.pkl', 'rb'))
delta_kpi = pickle.load(open('delta_kpi.pkl', 'rb'))
# evaluation_dict = pickle.load(open('evaluation_dict.pkl', 'rb'))
solutions_tree = pickle.load(open('solutions_tree.pkl', 'rb'))


def generate_choice_customized_prob(weights=[.45, .165, .115, .06, 0.045, .065, .025, .03, .015, .03, .02, .02]):
    return random.choices(range(0,12), weights)

def generate_case_resource_dict(solutions_tree):

    c =0
    resources_cases_dict = dict()
    for res in tqdm.tqdm(list(pm4py.get_event_attribute_values(log, "org:resource").keys())):
        resources_cases_dict[res] = dict()
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









    for case_name in tqdm.tqdm([i[0]['Case_id'].unique() for i in solutions_tree.values()]):
        for case_name2 in case_name:
            if case_name2 not in list(cases_resources_dict.keys()):
                cases_resources_dict[case_name2] = list()

    return cases_resources_dict




def evaluate_set(solutions_tree, act_role_dict_eval, customized=True):

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
            print(res_list.index(arrived_res))

            while chosen_case in cases_done:
                    #Generate its choiche between the given probability (paper referenced)
                    choice_idx = generate_choice_customized_prob()[0]
                    cases_set = set()
                    while len(cases_set)<=10:
                        idx_res += 1




                    # considered_solution = solutions_tree[idx+choice_idx][solutions_tree[idx]['Resource']==arrived_res]
                    # # considered_solution = solutions_tree[idx][solutions_tree[idx]['Resource']==arrived_res]
                    # act_sol = considered_solution['Activity_recommended'].values[0]
                    # chosen_case = considered_solution['Case_id'].values[0]

            cases_done.add(chosen_case)
            if choice_idx == 0:
                idx_res += 1
            cumulative_avg_kpi += act_role_dict_eval[act_sol][arrived_res]
            # print(f'the current score for case {chosen_case} and res {arrived_res} is {act_role_dict_eval[act_sol][arrived_res]}')


        return cumulative_avg_kpi


a = evaluate_set(solutions_tree=solutions_tree, act_role_dict_eval=act_role_dict_eval, customized=True)
# b = evaluate_set(solutions_tree=solutions_tree, act_role_dict_eval=act_role_dict_eval, customized=False)


if __name__ == '__main__':
    b = 'not implemented'
    print(f'Code executed the cumulative avg time is {a} against the time without our ranking is {b}')



