import sys
import pandas as pd
import io
import re
from tqdm import tqdm

if(len(sys.argv) == 1):
    print("Please provide a filepath")
    sys.exit()

with open(sys.argv[1],'r') as f:
    data = f.read()
df = pd.read_csv(io.StringIO(data))

targets = df['target_block'].to_list()
for i in range(len(targets)):
    targets[i] = targets[i].replace(' ','')

def mask_replace(i,input):
    regex = re.compile(' *'.join(map(re.escape, targets[i])))
    substring = regex.search(input)
    if substring:
        return input.replace(input[substring.start():substring.end()],'<IF-STMT>',1)
    else:
        print("Substring not found :(")

for i in tqdm(range(len(df))):
    df.at[i,'cleaned_method'] = mask_replace(i,df.at[i,'cleaned_method'])
    df.at[i,'cleaned_method'] = df.at[i,'cleaned_method'].replace('\n','')
    df.at[i,'cleaned_method'] = df.at[i,'cleaned_method'].replace('    ','<tab>')

output_filename = 'output.csv'
if(sys.argv[1] == '.\\Archive\\ft_test.csv'): output_filename = 'test_output.csv'
if(sys.argv[1] == '.\\Archive\\ft_train.csv'): output_filename = 'train_output.csv'
if(sys.argv[1] == '.\\Archive\\ft_valid.csv'): output_filename = 'valid_output.csv'

df.to_csv(output_filename)