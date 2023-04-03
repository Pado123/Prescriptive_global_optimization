import pickle
import numpy as np
import pandas as pd
import random
import tqdm
import os
import glob
from operator import itemgetter

random.seed(1618)

# Converter
import pm4py

# My functions
import hash_maps
import utils
import argparse


parser = argparse.ArgumentParser(
    description='Main script for Eval_System')

parser.add_argument('--cut', default=1)
parser.add_argument('--pad_mode', default=True)
args = parser.parse_args()

# mandatory parameters
cut = args.cut
pad_mode = args.pad_mode
cut = float(cut)

experiment_name = 'BACTIME'
case_id_name = 'REQUEST_ID'
pred_column = 'remaining_time' #'independent_activity'
res_name = 'CE_UO'

# experiment_name = 'BACTIME' #
# case_id_name = 'REQUEST_ID'
# pred_column = 'independent_activity'
# res_name = 'CE_UO'

if not os.path.exists(f'results_pgo'):
    os.mkdir('results_pgo')

X_train, X_test, y_train, y_test = utils.import_vars(experiment_name=experiment_name, case_id_name=case_id_name)
activity_name = 'concept:name'
try:
    df_rec = pickle.load(open(f'df_rec_{experiment_name}.pkl', 'rb'))
except:
    print('generating_dfrec')
    df_rec = utils.get_test(X_test, case_id_name).reset_index(drop=True)
    if 'ACTIVITY' in X_train.columns:
        df_rec.rename(columns={case_id_name: 'case:concept:name', 'ACTIVITY': 'concept:name', res_name : 'org:resource'},
                      inplace=True)
    pickle.dump(df_rec, open(f'df_rec_{experiment_name}.pkl', 'wb'))

columns = X_test.columns



if 'ACTIVITY' in X_train.columns:
    X_train.rename(columns={case_id_name: 'case:concept:name', 'ACTIVITY': 'concept:name', res_name : 'org:resource'})

if 'ACTIVITY' in X_test.columns:
    X_test.rename(columns={case_id_name: 'case:concept:name', 'ACTIVITY': 'concept:name', res_name: 'org:resource'},
                  inplace=True)

# case_id_name = 'case:concept:name'
# log = pm4py.convert_to_event_log(X_train)
# roles = pm4py.discover_organizational_roles(log)
# available_resources_list = list(pm4py.get_event_attribute_values(log, "org:resource").keys())
# activity_list = list(X_train['concept:name'].unique())
# act_role_dict = dict()
# cases_list = list(df_rec['case:concept:name'].unique())
# traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name, thrs=0)
# traces_hash = pickle.load(open('traces_hash.pkl', 'rb'))
# model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
# quantitative_vars = pickle.load(open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'rb'))
# qualitative_vars = pickle.load(open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'rb'))
# df_sol = pd.read_csv(f'Results_mixed_r_{experiment_name}.csv', index_col=0)
# df_sol['Case_id'] = df_sol['Case_id'].astype(str)
case_id_name = 'case:concept:name'
df_rec[case_id_name] = df_rec[case_id_name].astype(str)
delta_KPI = pickle.load(open(f'delta_kpi_{experiment_name}.pkl', 'rb'))
possible_permutations = dict()
initial_order = list(delta_KPI.keys())

# log = pm4py.convert_to_event_log(X_train)
# act_role_dict = pickle.load(open(f'act_role_dict_{experiment_name}.pkl', 'rb'))

delta_kpi = pickle.load(open(f'delta_kpi_{experiment_name}.pkl', 'rb'))


# del log, X_test, X_train, y_train, y_test, delta_kpi

def evaluate_resource_case_variability(solutions_tree):
    for res in solutions_tree[list(solutions_tree.keys())[0]][0]['Resource']:
        a = set()
        for key in solutions_tree.keys():
            try:
                df = solutions_tree[key][0]
                a.add(df[df['Resource'] == res]['Case_id'].values[0])
            except:
                None
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
            score = df_for_evaluation.loc[
                (df_for_evaluation['concept:name'] == act) & (df_for_evaluation['org:resource'] == res), [
                    'y']].mean().values[0]

            if act in act_role_dict_eval.keys():
                act_role_dict_eval[act][res] = score

    return act_role_dict_eval


def generate_choice_customized_prob(weights=[.45, .165, .115, .06, 0.045, .065, .025, .03, .015, .03, .02, .02]):
    return random.choices(range(0, 12), weights=weights)[0]


def generate_case_resource_dict(solutions_tree):  # 6 MIN

    c = 0
    resources_cases_dict = dict()
    res_list = set(solutions_tree[list(solutions_tree.keys())[0]][0]['Resource'].unique())
    for key in solutions_tree.keys():
        res_list = res_list.intersection(set(solutions_tree[key][0]['Resource'].unique()))
    res_list = list(res_list)
    for res in tqdm.tqdm(res_list):
        resources_cases_dict[res] = dict()
        if res != 'missing':
            for key in solutions_tree.keys():
                try:
                    case, score = \
                        solutions_tree[key][0][solutions_tree[key][0]['Resource'] == res][
                            ['Case_id', 'Expected KPI']].values[0]
                    if case in resources_cases_dict[res].keys():
                        resources_cases_dict[res][case].append(score)
                    if case not in resources_cases_dict[res].keys():
                        resources_cases_dict[res][case] = [score]

                except:
                    c += 1
                    print('baba', c)

    # Replace it with its associated KPI
    for res in tqdm.tqdm(resources_cases_dict.keys()):
        for case in resources_cases_dict[res]:
            resources_cases_dict[res][case] = np.mean(resources_cases_dict[res][case])
    for res in tqdm.tqdm(resources_cases_dict.keys()):
        resources_cases_dict[res] = {k: v for k, v in
                                     sorted(resources_cases_dict[res].items(), key=lambda item: item[1])}

    return resources_cases_dict

def evaluate_set(solutions_tree, delta_kpi, X_test, y_test, resources_cases_dict, customized=True,
                 pad_mode=True, predict_activities=None):

    # Get the resources of the tree and rename the keys for convenience
    res_list = set(solutions_tree[list(solutions_tree.keys())[0]][0]['Resource'].unique())
    for key in tqdm.tqdm(solutions_tree.keys()):
        res_list = res_list.intersection(set(solutions_tree[key][0]['Resource'].unique()))
    res_list = list(res_list)
    new_keys = range(len(solutions_tree.keys()))
    # res_list = pickle.load(open('res_list.pkl', 'rb'))

    # This has been removed for matching the function below
    # solutions_tree = dict(zip(new_keys, [i[0] for i in solutions_tree.values()]))

    # Random shuffle the resources for their arrival order (it is done in-place) #TODO: rimetti a posto queste cose
    random.shuffle(res_list)

    # Get the columns for the evaluation part
    columns = X_test.columns

    # Following their arrival order, evaluate the choice of the list
    cumulative_avg_kpi = 0
    cases_done = set()
    skipped = 0
    chosen = 0
    failed = 0
    selected_solutions = dict()
    lenghts = []

    if customized:
        if pad_mode:
            # Generate an id-resource dictionary for the pointwise situation (ordered by KPI value)
            # resources_cases_dict = generate_case_resource_dict(solutions_tree)
            for arrived_res in tqdm.tqdm(res_list):
                # print(f'the loop is at {res_list.index(arrived_res)}, and cases_done are {len(cases_done)}')
                chosen_case = None
                chosen_assigned = False
                while chosen_case not in cases_done:
                    try:
                        # Get the best-10 available cases and filter it with the already done cases
                        possible_cases = list(resources_cases_dict[arrived_res].keys())
                        if len(possible_cases) == 0:
                            print('case skipped')
                            skipped += 1
                        if len(possible_cases) > 1:
                            if not chosen_assigned:
                                chosen += 1
                                chosen_assigned = True

                        num_choice = 11  # Messo alto a caso per fare entrare sicuramente in while
                        while num_choice >= len(possible_cases):
                            print(f' lpc is {len(possible_cases)}')
                            num_choice = generate_choice_customized_prob()

                        # Choice the case
                        chosen_case = possible_cases[num_choice]
                        chosen_act = delta_kpi[chosen_case][0][1]
                        cases_done.add(chosen_case)
                    except:
                        skipped += 1

                selected_solutions[chosen_case] = [arrived_res, chosen_act]
                print(f'To the resource {arrived_res}, the selected case is {chosen_case}, skipped cases {skipped}')

            '''EVALUATION PART '''
            print('Creating evaluation dataframe')
            try:
                df_score = pickle.load(open(f'df_score_{experiment_name}.pkl', 'rb'))
                df_rec = pickle.load(open(f'df_rec_{experiment_name}.pkl', 'rb'))
            except:
                print('Create evaluation dictionaries')
                df_score = utils.create_eval_set(X_test, y_test, add_res=True) #.values
                df_rec = utils.get_test(X_test, case_id_name='case:concept:name').reset_index(drop=True)
                pickle.dump(df_score, open(f'df_score_{experiment_name}.pkl', 'wb'))
                pickle.dump(df_rec, open(f'df_rec_{experiment_name}.pkl', 'wb'))
                print('Evaluation dataframe created')
            selected_solutions = pd.DataFrame.from_dict(selected_solutions, orient='index',
                                                        columns=['Resource', 'Activity'])
            selected_solutions['Case_id'] = selected_solutions.index
            selected_solutions = selected_solutions.reset_index(drop=True)[['Case_id', 'Resource', 'Activity']]
            for row, line in selected_solutions.iterrows():
                scoring_df = df_score[df_score['org:resource'] == line['Resource']].iloc[:, 1:].values
                try:
                    acts = list(df_rec[df_rec[case_id_name] == int(line['Case_id'])][activity_name].values)
                except:
                    try:
                        acts = list(df_rec[df_rec[case_id_name] == line['Case_id']][activity_name].values)
                    except:
                        failed += 1
                        continue
                try:
                    score, avg_num_of_samples = utils.from_trace_to_score(acts, pred_column=pred_column,
                                                                          activity_name=activity_name,
                                                                          df_score=scoring_df, columns=X_test.columns,
                                                                          predict_activities=predict_activities,
                                                                          remove_loop_consideration=True)  # Not tested for predict_activities!=None
                except:
                     print('db')
                if score == None:
                    failed += 1
                    # print(f'Score has failed')
                else:
                    cumulative_avg_kpi += score
                lenghts.append(avg_num_of_samples)
            return cumulative_avg_kpi, round(chosen / len(res_list), 2), selected_solutions, failed, lenghts

        if not pad_mode:

            # solutions_tree = solutions_tree.values()
            solutions_tree = list({k: solutions_tree[k][0] for k in list(solutions_tree.keys())}.values())
            first_idx = 0
            selected_solutions = pd.DataFrame(columns=['Case_id', 'Activity', 'Resource'])
            # Generate an ordered-list of profiles
            for arrived_res in tqdm.tqdm(res_list):
                print(f'the loop is at {res_list.index(arrived_res)}, and cases_done are {len(cases_done)}')
                chosen_case = None
                num_choice = 11
                while num_choice + 1 > 3:
                    num_choice = generate_choice_customized_prob()
                print(f'num choice is {num_choice + 1}')
                chosen_assigned = False
                while chosen_case not in cases_done:

                    # Select the list of possible cases
                    possible_cases, first_idx = utils.get_possible_cases_maxtype(base_idx=first_idx,
                                                                                 resource=arrived_res,
                                                                                 solutions_tree=solutions_tree,
                                                                                 selected_solutions=selected_solutions,
                                                                                 cases_done=cases_done,
                                                                                 num_choice=num_choice)

                    if len(possible_cases) == 0:
                        print('case skipped')
                        skipped += 1
                        break
                    if len(possible_cases) > 1:
                        if chosen_assigned:
                            chosen += 1
                        try:
                            chosen_case = possible_cases[num_choice]
                        except:
                            num_choice -= 1
                            chosen_case = possible_cases[num_choice]
                    if len(possible_cases) == 1:
                        chosen_case = possible_cases[0]

                    print(f' lpc is {len(possible_cases)}')
                    # Choice the case

                    chosen_act = delta_kpi[chosen_case][0][1]
                    cases_done.add(chosen_case)
                selected_solutions = selected_solutions.append(
                    pd.Series(tuple((chosen_case, chosen_act, arrived_res)), index=selected_solutions.columns),
                    ignore_index=True)
            pickle.dump(selected_solutions, open('selected_solutions.pkl', 'wb'))
            # selected_solutions = pickle.load(open('selected_solutions.pkl', 'rb'))
            selected_solutions.rename(columns={'Activity_recommended': 'Activity'}, inplace=True)
            '''EVALUATION PART'''
            print('Creating evaluation dataframe')
            # df_score = utils.create_eval_set(X_test, y_test, add_res=True) #.values
            # df_rec = utils.get_test(X_test, case_id_name='case:concept:name').reset_index(drop=True)
            df_score = pickle.load(open(f'df_score_{experiment_name}.pkl', 'rb'))
            df_rec = pickle.load(open(f'df_rec_{experiment_name}.pkl', 'rb'))
            print('Evaluation dataframe created')
            # selected_solutions = pd.DataFrame.from_dict(selected_solutions, orient='index',
            #                                             columns=['Resource', 'Activity'])
            # selected_solutions['Case_id'] = selected_solutions.index
            selected_solutions = selected_solutions.reset_index(drop=True)[['Case_id', 'Resource', 'Activity']]
            for row, line in selected_solutions.iterrows():  
                scoring_df = df_score[df_score['org:resource'] == line['Resource']].iloc[:, 1:].values
                try:
                    acts = list(df_rec[df_rec[case_id_name] == int(line['Case_id'])][activity_name].values)
                except:
                    failed += 1
                    continue
                score, avg_num_of_samples = utils.from_trace_to_score(acts, pred_column=pred_column,
                                                                      activity_name=activity_name,
                                                                      df_score=scoring_df, columns=X_test.columns,
                                                                      predict_activities=predict_activities,
                                                                      remove_loop_consideration=True)


                if score == None:
                    failed += 1
                    print(f'Score has failed')
                else:
                    cumulative_avg_kpi += score
                lenghts.append(avg_num_of_samples)
            return cumulative_avg_kpi, round(chosen / len(res_list), 2), selected_solutions, failed, lenghts


def merge_sol(sol1, sol2):
    # Rename the keys of the dictionary for having just an order
    r1, r2 = range(len(sol1)), range(len(sol1), len(sol1) + len(sol2))
    sol1 = {i: sol1[k] for i, k in zip(r1, sol1.keys())}
    sol2 = {i: sol2[k] for i, k in zip(r2, sol2.keys())}

    # Merge and filter dictionaries
    sol1 = sol1 | sol2
    sol1 = utils.filter_and_reorder_solutions_dict(sol1, wise=True)
    return sol1
    
def cut_dict(diz, n=200):
    return {k: diz[k] for k in list(diz.keys())}


def merge_sol_all(experiment_name):
    # Set the directory in which there are the solutions
    try:
        os.chdir(f'trees_{experiment_name}')
    except:
        raise NotADirectoryError

    names = glob.glob('*')
    final_sol = dict()
    s1, s2 = pickle.load(open(names[0], 'rb')), pickle.load(open(names[1], 'rb'))
    final_sol = merge_sol(s1, s2)

    for name in glob.glob('*'):

        try:
            partial_solution = pickle.load(open(name, 'rb'))
        except:
            print(f'Error {name}')
        final_sol = merge_sol(final_sol, partial_solution)
        print(f'after merged {name} the solutions are {len(final_sol)}')

    os.chdir('..')

    return final_sol

if len(glob.glob(f'trees_{experiment_name}/*'))>1:
    solutions_tree = merge_sol_all(experiment_name)
else :
    name = glob.glob(f'trees_{experiment_name}/*')[0]
    solutions_tree = pickle.load(open(name, 'rb'))

# solutions_tree = pickle.load(open(f'solutions_tree_{experiment_name}.pkl', 'rb'))
# solutions_tree = {k: solutions_tree[k] for k in list(solutions_tree.keys())[:int(len(solutions_tree)*cut)]}

#Get random profiles
cut = int(cut*len(solutions_tree))
# random_indexes = random.sample(range(len(solutions_tree)), cut)
# chosen_keys = list(itemgetter(*random_indexes)(list(solutions_tree.keys())))
solutions_tree = {k: solutions_tree[k] for k in list(solutions_tree.keys())[:cut]}
try:
    resources_cases_dict = pickle.load(open(f'resources_cases_dict_{cut}_{experiment_name}.pkl', 'rb'))
except:
    print('generating rcd')
    resources_cases_dict = generate_case_resource_dict(solutions_tree)
    pickle.dump(resources_cases_dict, open(f'resources_cases_dict_{cut}_{experiment_name}.pkl', 'wb'))

avg_kpi_padmode, freedom_rate_padmode, selected_solutions_padmode, \
    failed_padmode, lenghts_padmode = evaluate_set(solutions_tree, delta_kpi, X_test, y_test, resources_cases_dict,
                                                   customized=True, pad_mode=True,
                                                   predict_activities=[
                                                       'Pending Liquidation Request'])
scores_dict = dict()
scores_dict[f'avg_kpi_padmode_{experiment_name}'] = avg_kpi_padmode
scores_dict[f'freedom_rate_{experiment_name}'] = freedom_rate_padmode
# scores_dict[f'selected_solutions_padmode_{experiment_name}'] = selected_solutions_padmode
scores_dict[f'failed_padmode_{experiment_name}'] = failed_padmode
scores_dict[f'lenghts_padmode_{experiment_name}'] = lenghts_padmode
pickle.dump(scores_dict, open(f'results_pgo/scores_dict_{pad_mode}_{str(cut)}_{experiment_name}.pkl', 'wb'))
try:
    df_score = pickle.load(open(f'df_score_{experiment_name}.pkl', 'rb'))
    df_rec = pickle.load(open(f'df_rec_{experiment_name}.pkl', 'rb'))
except:
    print('Create evaluation dictionaries')
    df_score = utils.create_eval_set(X_test, y_test, add_res=True)  # .values
    df_rec = utils.get_test(X_test, case_id_name='case:concept:name').reset_index(drop=True)
    pickle.dump(df_score, open(f'df_score_{experiment_name}.pkl', 'wb'))
    pickle.dump(df_rec, open(f'df_rec_{experiment_name}.pkl', 'wb'))
    print('Evaluation dataframe created')
scores_reality = utils.eval_base_value(selected_solutions = selected_solutions_padmode, df_rec=df_rec,
                                    df_score=df_score, case_id_name=case_id_name, activity_name=activity_name,
                                    pred_column=pred_column, X_test=X_test, predict_activities=['Pending Liquidation Request'])

resources_cannot_choose = utils.provide_freedom_value(resources_cases_dict, selected_solutions_padmode)
freedom_rate_reality = resources_cannot_choose/len(selected_solutions_padmode)

scores_reality = tuple(([i for i in scores_reality]+[freedom_rate_reality]))
pickle.dump(scores_reality, open(f'real_solutions_{pad_mode}_{cut}_{experiment_name}.pkl', 'wb'))
values_final = dict()
pad_accuracy = avg_kpi_padmode/(len(lenghts_padmode)-failed_padmode)
real_accuracy = scores_reality[0]/len(scores_reality[4])
values_final['accuracy improvement'] = 1-(real_accuracy - pad_accuracy ) / real_accuracy
values_final['freedom_ratio'] = freedom_rate_padmode / freedom_rate_reality
pickle.dump(values_final, open(f'final_comparison_{pad_mode}_{cut}_{experiment_name}.pkl', 'wb'))


if __name__ == '__main__':
    print(f'accuracy percentage improvement is {1 - (real_accuracy - pad_accuracy) / real_accuracy}')
    print(f'freedom ratio {freedom_rate_padmode / freedom_rate_reality}')
    print(f'We used {len(solutions_tree)} solutions')
