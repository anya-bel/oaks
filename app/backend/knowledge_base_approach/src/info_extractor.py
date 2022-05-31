from util import *
import json
from openie import StanfordOpenIE

# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

def extract_information(text_path):
    with StanfordOpenIE(properties=properties) as client:
        with open(text_path, encoding='utf8') as r:
            corpus = r.read().replace('\n', ' ').replace('\r', '')

        triples_corpus = client.annotate(corpus)
        print('Corpus: %s [...].' % corpus[0:80])
        print('Found %s triples in the corpus.' % len(triples_corpus))
        
        with open(data_path+'data/triples.txt','w') as file:
            for triple in triples_corpus:
                file.write(json.dumps(triple)+'\n')
        
        return triples_corpus