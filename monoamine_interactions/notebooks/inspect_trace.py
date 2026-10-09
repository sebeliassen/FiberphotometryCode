import pandas as pd
import numpy as np

df = pd.read_csv('/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/outputs/06_tabular_extraction/trace_landmarks.csv')
row = df[(df['Event'] == 'juvenile_first_sniff_min') & (df['Group'] == 'WT | lNAc_5HT -> rNAc_DA') & (df['Metric'] == 'Causality')]
print(row.to_string())

