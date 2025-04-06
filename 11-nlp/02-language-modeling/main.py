#!/home/malick/miniconda3/envs/pt/bin/python3

import sys
import utils
import markov_chains

def corpus_test():
    return ["<s> I am Sam </s>",
        "<s> Sam I am </s>",
        "<s> I do not like green eggs and ham </s>"]

def test_bigram_modeling():
    corpus = corpus_test()
    bag_of_words = utils.bag_of_ngrams(corpus=corpus, n=1, tokenize_function=utils.split_tokenize)
    print("Bag of words:\n", bag_of_words)

    bigram = utils.bag_of_ngrams(corpus=corpus, n=2, tokenize_function=utils.split_tokenize)
    print("Bigram:\n", bigram)

    def ngrams_compute_pw1_knowing_w2(w2, w1, bag_of_words, bigram):
        p = bigram[f"{w1} {w2}"].sum() / bag_of_words[w1].sum()
        print(f"p({w2}|{w1})={p:.4f}")

    ngrams_compute_pw1_knowing_w2(w2="I", w1="<s>", bigram=bigram, bag_of_words=bag_of_words)
    ngrams_compute_pw1_knowing_w2(w2="</s>", w1="Sam", bigram=bigram, bag_of_words=bag_of_words)
    ngrams_compute_pw1_knowing_w2(w2="eggs", w1="green", bigram=bigram, bag_of_words=bag_of_words)
    ngrams_compute_pw1_knowing_w2(w2="Sam", w1="am", bigram=bigram, bag_of_words=bag_of_words)
    ngrams_compute_pw1_knowing_w2(w2="am", w1="I", bigram=bigram, bag_of_words=bag_of_words)
    ngrams_compute_pw1_knowing_w2(w2="do", w1="I", bigram=bigram, bag_of_words=bag_of_words)


def test_markov_chains():
    corpus = corpus_test()
    tokenized_corpus, words_freq = utils.split_tokenize(corpus=corpus)

    states = list(words_freq.keys())
    markov_model = markov_chains.MarkovChains(states=states)

    print("\nStates:\n", markov_model.states, "\n")
    print("\nTokenized corpus:\n", tokenized_corpus, "\n")

    markov_model.fit(tokenized_corpus=tokenized_corpus)
    print("\nInitial probabilities:\n", markov_model.init_probs)
    print("\nTransition matrix:\n", markov_model.transition_matrix)

    print("\nMost probabale next states:")
    for current_state in markov_model.states:
        next_state, probas = markov_model.predict_next_state(current_state=current_state)
        print(f"Current state: [{current_state}]".ljust(23)
            + f"| next state: [{next_state}]".ljust(22)
            + f"| with proba: p({current_state}|{next_state})={probas.max():.4f}")

    print("\nGenerate sequences:")
    print("Seq1:", " ".join(markov_model.generate(start=None, length=10)))
    print("Seq2:", " ".join(markov_model.generate(start="<s>", length=10)))
    print("Seq3:", " ".join(markov_model.generate(start="I", length=10)))

    print("\nEvaluation:")
    data = [
        ("machine translation", "<s> I am", "<s> am I"),
        ("spell correction", "<s> I do not like", "<s> I do not lik"),
    ]

    for title, text1, text2 in data:
        score1 = markov_model.score_sequence(sequence=text1.split())
        score2 = markov_model.score_sequence(sequence=text2.split())
        print(f"{title.ljust(20)}: p({text1})={score1:.4f} > p({text2})={score2:.4f}")

if __name__ == "__main__":
    functions = [test_bigram_modeling, test_markov_chains]
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
