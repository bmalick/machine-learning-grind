#!/home/malick/miniconda3/envs/pt/bin/python3

import sys
from datasets import load_dataset
import matplotlib.pyplot as plt
from typing import List

import utils

def get_wolof_corpus() -> List[str]:
    raw_data = load_dataset("michsethowusu/afrikaans-wolof_sentence-pairs")["train"]
    print(raw_data)
    return raw_data["Wolof"]

def test_zipf_law():
    wolof_corpus = get_wolof_corpus()
    fig, ax = plt.subplots()
    utils.zipf_law(corpus=wolof_corpus, ax=ax)
    plt.show()

def test_bag_of_words():
    wolof_corpus = get_wolof_corpus()
    bag = utils.bag_of_words(corpus=wolof_corpus[:2500])
    print(bag)

def test_bag_of_ngrams():
    wolof_corpus = get_wolof_corpus()
    bag = utils.bag_of_ngrams(corpus=wolof_corpus[:2500], n=2)
    print(bag)

def test_tfidf():
    wolof_corpus = get_wolof_corpus()
    tfidf_corpus = utils.tfidf(wolof_corpus[:2500])
    print(tfidf_corpus)

if __name__ == "__main__":
    functions = [
        test_zipf_law, test_bag_of_words, test_bag_of_ngrams, test_tfidf]
    if len(sys.argv) !=2:
        print("Usage: %s <function id>" % sys.argv[0])
        print()
        print("id | function")
        print("---+"+'-'*20)
        for id, f in enumerate(functions):
            print("%d  | %s" %(id, f.__name__))
        sys.exit()

    id = int(sys.argv[1])
    if(id < 0 or id >= len(functions)) :
        print("Function id %d is invalid (should be in [0, %d])" % (id, len(functions)-1))
        sys.exit()
    functions[id]()

