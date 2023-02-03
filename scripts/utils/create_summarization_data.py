import json
import sys
import stanza
nlp = stanza.Pipeline('en', package='craft')

data = json.load(open(sys.argv[1], "r"))
silver_labels = {}
for record in data:
    doi = record["doi"]
    sents = [sent.text for sent in nlp(record["abstract"]).sentences]
    labels = record["abstract_sentence_label"]
    silver_labels[doi] = (sents, labels)

for split in ["train", "val", "test"]:
    doi_file = open(f'data/data-1024/{split}.doi', 'r').readlines()
    data_file = open(f'data/data-summarization/{split}.txt', 'w')
    data_file.write("doi\tsent\tlabel\n")
    for doi in doi_file:
        sents, labels = silver_labels[doi.strip()]
        for sent, label in zip(sents, labels):
            data_file.write("%s\t%s\t%s\n"%(doi.strip(), sent, label))
    data_file.close()
