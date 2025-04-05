import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

def tokenize(corpus: List[str]) -> Tuple[List[List[str]], Dict[str, int]]:
    words_freq = {}
    tokens = []
    for sentence in corpus:
        if sentence:
            res = re.findall(r"\w+", sentence.lower())
            tokens.append(res)
            for w in res:
                words_freq[w] = words_freq.get(w, 0) + 1
    return tokens, words_freq

def zipf_law(corpus: List[str], ax):
    _, words_freq = tokenize(corpus=corpus)
    words_freq = dict(sorted(words_freq.items(), key=lambda x: x[1], reverse=True))

    log_freqs = np.log(np.array(list(words_freq.values())))
    ranks = np.arange(1, len(log_freqs)+1)

    ax.loglog(ranks, log_freqs)
    ax.set_xlabel("rank")
    ax.set_ylabel("frequency")
    return words_freq

def get_counts_from_tokens(tokens: List[str]) -> Dict[str, int]:
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts

def bag_of_words(corpus: List[str], vocab_size: int=None) -> pd.DataFrame:
    tokenized_corpus, words_freq = tokenize(corpus=corpus)
    n_docs = len(tokenized_corpus)

    if vocab_size is None: vocab_size = len(words_freq)
    vocab = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = [w for w,_ in vocab[:vocab_size]]

    bag = pd.DataFrame(data=0., index=range(n_docs), columns=vocab)

    for doc_id, tokens in tqdm(enumerate(tokenized_corpus), desc="bag of words"):
        term_counts = get_counts_from_tokens(tokens=tokens)
        for term in vocab:
            if term in term_counts:
                bag.loc[doc_id, term] = term_counts[term]
    return bag

def get_ngrams(tokenized_corpus: List[List[str]], n: int) -> Tuple[List[List[str]], Dict[str, int]]:
    result = []
    ngrams_freq = {}
    for tokens in tqdm(tokenized_corpus, desc=f"{n}grams"):
        ngrams = []
        for i in range(len(tokens)-n+1):
            text = " ".join(tokens[i:i+n])
            ngrams_freq[text] = ngrams_freq.get(text, 0) + 1
            ngrams.append(text)
        result.append(ngrams)
    return result, ngrams_freq

def bag_of_ngrams(corpus: List[str], n: int=2, vocab_size: int=None) -> pd.DataFrame:
    tokenized_corpus, _= tokenize(corpus=corpus)
    ngrams_corpus, ngrams_freq = get_ngrams(tokenized_corpus=tokenized_corpus, n=n)

    if vocab_size is None: vocab_size = len(ngrams_freq)
    vocab = sorted(ngrams_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = [w for w,_ in vocab[:vocab_size]]

    n_docs = len(ngrams_corpus)
    bag = pd.DataFrame(data=0., index=range(n_docs), columns=vocab)

    for doc_id, tokens in tqdm(enumerate(ngrams_corpus), desc=f"bag of {n}grams"):
        term_counts = get_counts_from_tokens(tokens=tokens)
        for term in vocab:
            if term in term_counts:
                bag.loc[doc_id, term] = term_counts[term]
    return bag

def tfidf(corpus: List[str], vocab_size: int=None) -> pd.DataFrame:
    tokenized_corpus, words_freq = tokenize(corpus=corpus)

    if vocab_size is None: vocab_size = len(words_freq)
    vocab = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = [w for w,_ in vocab[:vocab_size]]

    n_docs = len(tokenized_corpus)
    result = pd.DataFrame(0., index=range(n_docs), columns=vocab)

    # document frequencies
    doc_freq = {term: 0 for term in vocab}
    for doc in tokenized_corpus:
        unique_terms = set(doc)
        for term in unique_terms:
            if term in doc_freq:
                doc_freq[term] += 1

    # inverse document frequency
    idf = {term: np.log(n_docs / freq) if freq>0 else 0.
           for term, freq in doc_freq.items()}

    for doc_id, doc in tqdm(enumerate(tokenized_corpus), desc="tfidf"):
        doc_length = len(doc)
        term_counts = get_counts_from_tokens(tokens=doc)

        for term in vocab:
            if term in term_counts:
                # term frequency: tf = count / document_length
                tf = term_counts[term] / doc_length
                # tf-idf = tf * idf
                result.loc[doc_id, term] = tf * idf[term]

    return result
