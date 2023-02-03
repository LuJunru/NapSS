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
             "val": "data/data-summarization/val.txt", 
             "test": "data/data-summarization/test.txt"}
dataset = load_dataset('csv', data_files=file_dict, delimiter='\t', column_names=['doi', 'sent', 'label'], skiprows=1)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("data/data-summarization/%s-sents"%model_name, num_labels=2)
training_args = TrainingArguments(
   output_dir="data/data-summarization/checkpoints",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=False,
   no_cuda=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

import json
import stanza
nlp = stanza.Pipeline('en', package='craft')
data = json.load(open('data/data_final_1024-keyswithlabels.json', 'r'))

for split in ['val', 'test']:
    predictions = trainer.predict(tokenized_datasets[split])[0].argmax(-1)
    prediction_dict = {}
    for r, p in zip(tokenized_datasets[split], predictions):
        r_doi = r["doi"]
        if r_doi not in prediction_dict:
            prediction_dict[r_doi] = [p]
        else:
            prediction_dict[r_doi].append(p)

    data_dict = {}
    for record in data:
        if record["doi"] in prediction_dict:
            abs_label = prediction_dict[record["doi"]]
        else:
            abs_label = record["abstract_sentence_label"]
        data_dict[record["doi"]] = (record["abstract_keys"], abs_label)

    doi_file = open(f'data/data-1024-napss/{split}.doi', 'r').readlines()
    source_file = open(f'data/data-1024/{split}.source', 'r').readlines()
    new_source_file = open(f'data/data-1024-napss/{split}.source', 'w')

    for doix, doi in enumerate(doi_file):
        doc = nlp(source_file[doix].strip())
        s_abstract_sents = []
        for k, l, s in zip(data_dict[doi.strip()][0], data_dict[doi.strip()][1], doc.sentences):
            if l == 1:
                s_abstract_sents.append(s.text)
        source_file[doix] = " </s> ".join(data_dict[doi.strip()][0]) + " </s> " + " ".join(s_abstract_sents) + "\n"
        new_source_file.write(source_file[doix])

    new_source_file.close()

