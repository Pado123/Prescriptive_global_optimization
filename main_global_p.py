import pickle
import time
import numpy as np
import pandas as pd
import tqdm
import random
random.seed(1618)

# Converter
import pm4py

#My functions
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
available_resources_list = list(pm4py.get_event_attribute_values(log, "org:resource").keys())
activity_list = list(X_train['concept:name'].unique())
act_role_dict = dict()
cases_list = list(df_rec['case:concept:name'].unique())
# for act in activity_list:
#     for idx in range(len(roles)):
#         if act in roles[idx][0]:
#             act_role_dict[act] = list(roles[idx][1].keys())
# pickle.dump(act_role_dict, open('act_role_dict.pkl', 'wb'))
act_role_dict = pickle.load(open('act_role_dict.pkl', 'rb'))
traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name,
                                     thrs=0)
# traces_hash = pickle.load(open('traces_hash.pkl', 'rb'))

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

#Filter the resources for keeping just the active resources, IT ALSO REMOVES MISSING
available_resources_list = utils.filter_resources_availability(available_resources_list, p=.75)
if 'missing' in available_resources_list: available_resources_list.remove('missing')

#Associate prediction to KPI
delta_KPI = dict()
c=0
print('Starting generating Delta_Kpi')
for case_id in tqdm.tqdm(cases_list):
    trace = df_rec[df_rec[case_id_name] == case_id].reset_index(drop=True)
    last = trace.loc[max(trace.index)].copy()

    try:
        next_activities, actual_prediction = next_act.next_act_kpis(trace, traces_hash, model, pred_column,
                                                                    case_id_name,
                                                                    activity_name,
                                                                    quantitative_vars, qualitative_vars,
                                                                    encoding='aggr-hist')
        next_activities.sort_values(by='kpi_rel', inplace=True)
        next_activities.reset_index(drop=True, inplace=True)
        delta_KPI[str(case_id)] = list()
        for line in next_activities.index:
            delta_KPI[str(case_id)].append((actual_prediction - next_activities.iloc[line]['kpi_rel'], next_activities.iloc[line]['Next_act']))
    except:
        c+=1
print(f'The number of missed cases is {c}, and the final time {time.time()}') #10% of missed cases, just 234 different prediction values, maybe the predictor is too much lowered
delta_KPI = dict(sorted(delta_KPI.items(), key=lambda item: item[1][0][0], reverse=True))
keys_order = {k:v[0][0] for k,v in delta_KPI.items()}
keys_order = dict(sorted(keys_order.items(), key=lambda x: x[1], reverse=True))
delta_KPI = {k:delta_KPI[k] for k in keys_order}
# delta_KPI = pickle.load(open('delta_kpi.pkl', 'rb'))

Sol = list()
df_rec['case:concept:name'] = [str(i) for i in df_rec['case:concept:name']]

#Time counter for loop
delta_KPI = pickle.load(open('delta_kpi.pkl', 'rb'))
print('Starting generating the first solution')
for trace_idx in tqdm.tqdm(list(delta_KPI.keys())):

    appended = False #variable inserted for understanding if exit from one of two loops
    if available_resources_list == set():
        print('Resources finished')
        break

    #Select the best activity for the case
    delta_KPIa = np.array(delta_KPI[trace_idx])
    for i in range(len(delta_KPIa)):

        if appended == True:
            continue

        act_recommended = delta_KPIa[i][1]
        try:
            resources_for_act = act_role_dict[act_recommended]
        except:
            resources_for_act = None
        if set(resources_for_act).intersection(available_resources_list) == set():
            # print(f'No resource currently available for case {trace_idx}')
            continue

        #Select the case and include the current act as traces' act
        pred_case = df_rec[df_rec[case_id_name]==trace_idx]
        last = pred_case.loc[max(pred_case.index)].copy()
        last[activity_name] = act_recommended

        # put in all the columns which are not inferrable a null value
        for var in last.index:
            if var in (set(quantitative_vars).union(qualitative_vars)):
                last[var] = "none"

        partial_results = dict()
        resources_for_act = set(resources_for_act).intersection(available_resources_list)
        if resources_for_act == set():
            print(f'b2')
            continue

        for res in resources_for_act:
            last['org:resource'] = res
            # Create a vector with the actual prediction
            if pred_column == 'remaining_time':
                actual_prediction = model.predict(list(last[1:]))
            elif pred_column == 'independent_activity':
                actual_prediction = model.predict_proba(list(last[1:]))[0]  # activity case
            partial_results[res] = actual_prediction

        partial_results = dict(sorted(partial_results.items(), key=lambda item: item[1], reverse=False))
        best_res, expected_KPI  = list(partial_results.keys())[0], list(partial_results.values())[0]
        Sol.append((trace_idx, act_recommended, best_res, expected_KPI))
        available_resources_list.remove(best_res)
        appended = True
        print(f'Got sol for case {trace_idx}')

        continue
    continue

pickle.dump(Sol, open('Sol.pkl', 'wb'))
pickle.dump(delta_KPI, open('delta_kpi.pkl', 'wb'))
df_sol = pd.DataFrame(Sol, columns=['Case_id', 'Activity_recommended', 'Resource', 'Expected KPI'])
df_sol.to_csv('Results_mixed_r.csv')
print(f'THE FINAL TIME IS {time.time()}')



if __name__ == '__main__':
    print('ihihih')