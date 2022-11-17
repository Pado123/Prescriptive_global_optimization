import pickle
import pandas as pd
import pm4py

import tqdm
import hash_maps
import utils
from hash_maps import str_list, list_str
import next_act

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
model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
quantitative_vars = pickle.load(open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'rb'))
qualitative_vars = pickle.load(open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'rb'))

df_sol = pd.read_csv('Results_mixed_r.csv', index_col=0)
df_sol.columns = ['Case_id', 'Activity_recommended', 'Resource', 'Expected KPI']
df_sol['Case_id'] = df_sol['Case_id'].astype(str)
df_rec[case_id_name] = df_rec[case_id_name].astype(str)
delta_KPI = pickle.load(open('delta_kpi.pkl', 'rb'))

possible_permutations = dict()

def generate_permutation(df_sol, idx_resource, delta_KPI):

    #Get the case id and the free resource
    case = df_sol.iloc[idx_resource]
    case_id = case['Case_id']
    res_available = case['Resource']

    #Get the trace and prepare trace for prediction
    trace = df_rec[df_rec[case_id_name] == case_id].reset_index(drop=True)
    last = trace.loc[max(trace.index)].copy()
    # last[activity_name] = case['Activity_recommended']
    for var in last.index:
        if var in (set(quantitative_vars).union(qualitative_vars)):
            last[var] = "none"

    #Select the best activity
    next_activities, actual_prediction = next_act.next_act_kpis(trace, traces_hash, model, pred_column, case_id_name,
                                                                activity_name,
                                                                quantitative_vars, qualitative_vars,
                                                                encoding='aggr-hist')

    next_activities.sort_values(by=['kpi_rel'], inplace=True)
    best_new_act = next_activities['Next_act'][0]
    new_busy_resource = act_role_dict[best_new_act][0]


    # #Get the second best resource for activity and case
    # resources_for_act = act_role_dict[case['Activity_recommended']]
    # res_kpi = dict()
    # for res in resources_for_act:
    #     last['org:resource'] = res
    #     res_kpi[res] = model.predict(list(last)[1:])
    #
    # res_kpi = dict(sorted(res_kpi.items(), key=lambda item: item[1], reverse=False))
    # res_for_first = list(res_kpi.keys())[1] #1 because i want the 2nd best for the case

    #Get the trace for which the res will be removed and remove it from the df sol
    index_removed = int(df_sol[df_sol['Resource'] == res_for_first].index[0])
    case_removed = df_sol.iloc[index_removed]['Case_id']
    df_sol.drop(index=index_removed, inplace=True)
    df_sol.reset_index(drop=True, inplace=True)
    cases_id_already_used = set(df_sol['Case_id'].unique())
    print(f'Removed case {case_removed}')

    #Get the best case,activity given the resource removed
    cases_remainings = [case_id for case_id in delta_KPI.keys() if case_id not in cases_id_already_used]

    for case_id in tqdm.tqdm(cases_remainings):
        # Get the trace and prepare trace for prediction
        trace = df_rec[df_rec[case_id_name] == case_id].reset_index(drop=True)
        last = trace.loc[max(trace.index)].copy()
        trace_acts = list(trace[activity_name])
        # trace = trace[[col for col in trace.columns if col!='REQUEST_ID']]
        next_acts = traces_hash.get_val(str_list(trace_acts))
        if next_acts == 'No record found':
            # raise NotADirectoryError('Activity missed in hash-table')
            print(f'{case_id} è sbagliata non la trovo')
        for var in last.index:
            if var in (set(quantitative_vars).union(qualitative_vars)):
                last[var] = "none"



        for res in resources_for_act:
            last['org:resource'] = res
            res_kpi[res] = model.predict(list(last)[1:])

