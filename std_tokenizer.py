from collections import Counter
from nltk.corpus import stopwords


def build_standard_vocabulary(queries,documents,min_occ = 2):
    vocabulary = Counter()
    
    count = 0
    for _,document in documents.iterrows():
        for word in document[0].split(" "):
            vocabulary[word] += 1
        count +=1
       
    count = 0
    for _,query in queries.iterrows():
        for word in query[0].split(" "):
            vocabulary[word] += 1
        count +=1
    
    vocabulary = {i:elem[0] for i,elem in enumerate(vocabulary.most_common()) if elem[1] >= min_occ}
    
    for key in list(vocabulary):
        vocabulary[vocabulary[key]] = key
    
    return vocabulary


def index(pdDataFrame,vocabulary,stemmer=None):
    indexed_elements = []
    index = dict()
    count = 0
    if stemmer is None:
        for key,element in pdDataFrame.iterrows():
            indexed_elements.append([vocabulary[elem.lower()] for elem in element[0].split(" ") if elem.lower() in vocabulary])
            index[str(key)] = count
            index[count] = str(key)
            count += 1
            
    else:
        for key,element in pdDataFrame.iterrows():
            indexed_elements.append([vocabulary[stemmer.stem(elem.lower())] for elem in element[0].split(" ") if stemmer.stem(elem.lower()) in vocabulary])
            index[str(key)] = count
            index[count] = str(key)
            count += 1

    return index,indexed_elements



def index_dict(pdDataFrame,vocabulary):
    indexed_elements = []
    index = dict()
    count = 0
    for key,element in pdDataFrame.items():
        indexed_elements.append([vocabulary[elem] for elem in element.split(" ") if elem in vocabulary])
        index[str(key)] = count
        index[count] = str(key)
        count += 1
    return index,indexed_elements


def preprocess(queries,documents,min_occ = 5):
    
    vocabulary = build_standard_vocabulary(queries,
                                           documents,
                                           min_occ = min_occ)
    
    doc_index,indexed_docs = index(documents,vocabulary)
    
    query_index,indexed_queries = index(queries,vocabulary)
    
    return vocabulary,query_index,indexed_queries,doc_index,indexed_docs
    
