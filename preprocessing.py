import sys
import pandas as pd
import io

if(len(sys.argv) == 1):
    print("Please provide a filepath")
    sys.exit()

with open(sys.argv[1],'r') as f:
    data = f.read()
df = pd.read_csv(io.StringIO(data))

targets = df['target_block'].to_list()

for i in range(len(df)):
    df.at[i,'cleaned_method'] = df.at[i,'cleaned_method'].replace('\n','')
    #df.at[i,'cleaned_method'] = df.at[i,'cleaned_method'].replace(targets[i],'<mask>',1)
    df.at[i,'cleaned_method'] = df.at[i,'cleaned_method'].replace('    ','<tab>')

df.to_csv("output.csv")
