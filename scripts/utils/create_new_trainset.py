import json
import stanza
import sys
nlp = stanza.Pipeline('en', package='craft')

data = json.load(open(sys.argv[1], 'r'))
data_dict = {}
for record in data:
    data_dict[record["doi"]] = (record["abstract_keys"], record["abstract_sentence_label"])

for split in ['train']:
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
