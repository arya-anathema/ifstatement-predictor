#! pip install transformers
#!pip install tree_sitter==0.2.0
#! git clone -q https://github.com/microsoft/CodeXGLUE.git
#pip install sacrebleu

import subprocess
import sacrebleu

from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import pandas as pd


#test strings
pred1 = '@ Override public boolean removeRegistrationByUsername ( String username , CredentialRegistration credentialRegistration ) { try { return storage . get ( username , HashSet :: new ) . remove ( credentialRegistration ) ; } catch ( ExecutionException e ) { logger . error ( "Registration lookup failed" , e ) ; throw new RuntimeException ( e ) ; } }'
actu1 = '@ Override public boolean removeRegistrationByUsername ( String username , CredentialRegistration credentialRegistration ) { try { return storage . get ( username , HashSet :: new ) . remove ( credentialRegistration ) ; } catch ( ExecutionException e ) { logger . error ( "Failed to remove registration" , e ) ; throw new RuntimeException ( e ) ; } }'

def exactMatch(str1,str2):
  if(str1==str2):
    return 1
  else:
    return 0
  
def calcCodeBleu(str1,str2):
  with open("/content/actual_temp.txt", "w") as f:
    f.write(str1)

  with open("/content/pred_temp.txt", "w") as f:
    f.write(str2)

  command = """
cd /content/CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/ && python calc_code_bleu.py --refs /content/actual_temp.txt --hyp /content/pred_temp.txt --lang java --params 0.25,0.25,0.25,0.25"""
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  #print(result.stdout)

  output_lines = result.stdout.split('\n')
  score = 0
  for line in output_lines:
      if "CodeBLEU score" in line:
          score = float(line.split("CodeBLEU score:")[-1].strip())
          break
  return score

def calcSacreBlue(str1,str2):
  predlist = [str1]
  actulist = [[str2]]
  bleu = sacrebleu.corpus_bleu(predlist,actulist)

  return bleu.score

def calcF1Score(predStr,actuStr):

  true_strings = predStr.split()
  predicted_strings = actuStr.split()

  tokenizer = Tokenizer(num_words=None, char_level=False)
  tokenizer.fit_on_texts(true_strings + predicted_strings)

  true_sequences = tokenizer.texts_to_sequences([true_strings])
  predicted_sequences = tokenizer.texts_to_sequences([predicted_strings])

  max_length = max(len(true_strings), len(predicted_strings))
  padded_true_sequences = pad_sequences(true_sequences, maxlen=max_length, padding='post')
  padded_predicted_sequences = pad_sequences(predicted_sequences, maxlen=max_length, padding='post')

  f1_sequence = f1_score(padded_true_sequences.flatten(), padded_predicted_sequences.flatten(), average='macro')
  return f1_sequence


def preprocess_function(examples):
  inputs = examples["cleaned_method"]
  targets = examples["target_block"]
  model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
  labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


def dataset_init():
  valid_df = pd.read_csv('output_csv/valid_output.csv')
  valid_dataset = Dataset.from_pandas(valid_df)

  dataset = DatasetDict({
      'validation': valid_dataset
  })

  return dataset


model_checkpoint = "/content/model/checkpoint-100000"

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

dataset = dataset_init()

tokenized_datasets = dataset.map(preprocess_function, batched=True)

def getOutputString(input_method):

  inputs = tokenizer.encode(input_method, return_tensors="pt")

  output_ids = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)

  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

  return output_text

model_outputs = []

for i in range(len(tokenized_datasets['validation']['cleaned_method'])):
  if(i%10==0):
    print(i)
  model_outputs.append(getOutputString(tokenized_datasets['validation']['cleaned_method'][i]))

def compare(input, actualStr, predStr):
  results = []
  results.append(input)
  results.append(exactMatch(actualStr,predStr))
  results.append(actualStr)
  results.append(predStr)
  results.append(calcCodeBleu(actualStr,predStr))
  results.append(calcSacreBlue(predStr,actualStr))
  results.append(calcF1Score(predStr,actualStr))
  return results


print(compare(tokenized_datasets['validation']['cleaned_method'][0],tokenized_datasets['validation']['target_block'][0],model_outputs[0]))


resultsList = []
for i in range(len(model_outputs)):
  if(i%10==0):
    print(i)
  resultsList.append(compare(tokenized_datasets['validation']['cleaned_method'][i],tokenized_datasets['validation']['target_block'][i],model_outputs[i]))

print(resultsList)

df = pd.DataFrame(resultsList, columns=["input function", "match?", "expected", "predicted", "codeBleu", "Bleu-4", "F1"])
df.to_csv("results.csv", index=False)