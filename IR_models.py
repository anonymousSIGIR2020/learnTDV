import utils
import numpy as np
from collections import Counter

def simple_tf(indexed_queries,inverted_index):
    
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index:
                for document,freq in inverted_index[token].items():
                    result[document] += freq
        results.append(result)
        
    return results


def tf_idf(indexed_queries,inverted_index,idf):
    
    results = []
    
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index:
                for document,freq in inverted_index[token].items():
                    result[document] += freq*idf[token]
        results.append(result)
        
    return results

    
def dir_language_model(indexed_queries,inverted_index,docs_length,c_freq, mu = 2500):
    
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index:
                for document,freq in inverted_index[token].items():
                    result[document] += np.log(1 + (freq/(mu * c_freq[token]))) + np.log(mu/(docs_length[document] + mu))
        results.append(result)
        
    return results
    
    
def BM25(indexed_queries,inverted_index,docs_length,idf,avg_docs_len, k1 = 1.2, b = 0.75):
    
    results = []
        
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index:
                for document,freq in inverted_index[token].items():
                    result[document] += idf[token] * ((k1 + 1)*freq)/(freq + k1*((1-b) + b*docs_length[document]/avg_docs_len))
        results.append(result)
        
    return results
