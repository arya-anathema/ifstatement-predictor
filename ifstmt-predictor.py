from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import pandas as pd

def model_init():
    model_checkpoint = "Salesforce/codet5-small"

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_tokens(["<IF-STMT>"])

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

if __name__ == "__main__":

    model, tokenizer = model_init()
