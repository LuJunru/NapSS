import json
import nltk
import sys
import stanza
nlp = stanza.Pipeline('en', package='craft')

data = json.load(open(sys.argv[1], "r"))
for record in data:
    abs_keys = nlp(record["abstract"]).sentences
    pls_keys = nlp(record["pls"]).sentences
    record["abstract_sentence_label"] = [0] * len(abs_keys)
    if len(pls_keys) >= len(abs_keys):
        record["abstract_sentence_label"] = [1] * len(abs_keys)
    else:
        for pls_key in pls_keys:
            min_dist, min_dist_i = 100000, 0
            for abs_i, abs_key in enumerate(abs_keys):
                edit_dist = nltk.jaccard_distance(set(pls_key.text.split(" ")), set(abs_key.text.split(" ")))
                if edit_dist < min_dist:
                    min_dist = edit_dist
                    min_dist_i = abs_i
            if min_dist < 100000:
                record["abstract_sentence_label"][min_dist_i] = 1
json.dump(data, open(sys.argv[2], "w"))
