# ifstatement-predictor

### TODO:
- Flatten/mask preproccessed methods
  - Flatten: remove newlines
  - Mask: for each method, replace the if statement in the target column with <mask>, for methods with multiple of the same if statement, mask only the first
- Tokenize with given tokenizer in the T5 jupyter notebook
- Finetune model with 7-10 epochs
- Include evaluation metrics (CodeBLEU score, BLEU-4 score, exact match true/false, F1 score)
  - Store results in csv file: testset-results.csv
