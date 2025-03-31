from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import pandas as pd

#init
model_checkpoint = "Salesforce/codet5-small"

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<IF-STMT>"])
tokenizer.add_tokens(["<tab>"])

model.resize_token_embeddings(len(tokenizer))


def preprocess_function(examples):
    inputs = examples["cleaned_method"]
    targets = examples["target_block"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def dataset_init():
    test_df = pd.read_csv('output_csv/test_output.csv')
    train_df = pd.read_csv('output_csv/train_output.csv')
    valid_df = pd.read_csv('output_csv/valid_output.csv')

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })
    
    return dataset

if __name__ == "__main__":

    dataset = dataset_init()

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    print(tokenized_datasets['train']['target_block'][0])

    training_args = TrainingArguments(
        output_dir="./codet5-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_steps=100,
        push_to_hub=False,
        resume_from_checkpoint=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    metrics = trainer.evaluate(tokenized_datasets["test"])
    print("Test Evaluation Metrics:", metrics)
