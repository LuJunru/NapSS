import json
import sys
import stanza
# stanza.download('en', package='craft')
nlp = stanza.Pipeline('en', package='craft')
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

data = json.load(open(sys.argv[1], "r"))

for ri, record in enumerate(data):
    paragraph = record["abstract"]
    doc = nlp(paragraph)
    para_key_output = []
    for sentence in doc.sentences:
        srl = sentence.words
        for word in srl:
            if word.head == 0:
                srl_head = word.text
                srl_head_id = word.id
                break
        key_output = []
        for word in srl:
            if (word.head == srl_head_id or word.head == 0) and word.pos not in ["PUNCT", "SYM"]:
                key_output.append(word.text)
        para_key_output.append(" ".join(key_output))
    data[ri]["abstract_keys"] = para_key_output

    paragraph = record["pls"]
    doc = nlp(paragraph)
    para_key_output = []
    for sentence in doc.sentences:
        srl = sentence.words
        for word in srl:
            if word.head == 0:
                srl_head = word.text
                srl_head_id = word.id
                break
        key_output = []
        for word in srl:
            if (word.head == srl_head_id or word.head == 0) and word.pos not in ["PUNCT", "SYM"]:
                key_output.append(word.text)
        para_key_output.append(" ".join(key_output))
    data[ri]["pls_keys"] = para_key_output

json.dump(data, open(sys.argv[2], "w"))
