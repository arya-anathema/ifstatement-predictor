#! pip install transformers
#!pip install tree_sitter==0.2.0
#! git clone -q https://github.com/microsoft/CodeXGLUE.git
#pip install sacrebleu

import subprocess
import sacrebleu

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
    f.write(actu1)

  with open("/content/pred_temp.txt", "w") as f:
    f.write(pred1)

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

def calcF12Score(predstr1,actustr2):
  #F1 = TP / (TP +0.5(FP+FN))
  predictedTokens = set(predstr1.split())
  actualTokens = set(actustr2.split())

  TP = len(predictedTokens & actualTokens)
  FP = len(predictedTokens - actualTokens)
  FN = len(actualTokens - predictedTokens)

  F1 = TP / (TP+0.5*(FP+FN))
  return F1