import argparse
import pandas as pd
import pm4py

parser = argparse.ArgumentParser(
    description='Main script for Catboost training')

parser.add_argument('--filename')
parser.add_argument('--thrs', default=.5)
args = parser.parse_args()
filename = args.filename
thrs = args.thrs
thrs = float(thrs)

log = pm4py.read_xes(filename)
df = pm4py.convert_to_dataframe(log)
df_cut = pd.DataFrame(columns=df.columns)
for idx in df['case:concept:name'].unique():
    trace = df[df['case:concept:name']==idx].reset_index(drop=True)
    trace = trace.iloc[:round(len(trace)*thrs)]
    df_cut = df_cut.append(trace)
import ipdb; ipdb.set_trace()
log = pm4py.convert_to_event_log(df_cut)
pm4py.write_xes(log, f'{filename}_cut_{str(thrs)}_.xes')
