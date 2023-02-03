from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model_name = "michiyasunaga/BioLinkBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
import numpy as np
from datasets import load_metric
from datasets import load_dataset

import os
os.environ["WANDB_DISABLED"] = "true"

def tokenize_function(examples):
    return tokenizer(examples["sent"], truncation=True)

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

file_dict = {"train": "data/data-summarization/train.txt", 
             "valid": "data/data-summarization/val.txt", 
             "test": "data/data-summarization/test.txt"}
dataset = load_dataset('csv', data_files=file_dict, delimiter='\t', column_names=['doi', 'sent', 'label'], skiprows=1)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
training_args = TrainingArguments(
   output_dir="data/data-summarization/checkpoints",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=1,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=False,
   no_cuda=True,
   evaluation_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42),
    eval_dataset=tokenized_datasets["valid"].shuffle(seed=42),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(results)

trainer.save_model("data/data-summarization/%s-sents"%model_name)
