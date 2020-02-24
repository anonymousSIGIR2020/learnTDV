import os
import utils
import pickle
import random
import string
import fasttext
import numpy as np
import pytrec_eval
import pandas as pd
import std_tokenizer
from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords

#As the TREC collections are not publicly available, we removed the parts of the code that read and process the original files.

class TrecCollection:
    def __init__(self,k=5,language='english'):
        self.documents = None
        self.k = k
        self.language = language
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

        
    def load_collection(self,collection_path):
        self.documents = pd.read_csv(collection_path + '/documents.csv',index_col='id_right',na_filter=False)
        
        self.folds_queries = []
        self.folds_qrels = []
        self.folds_training_qrels = []
        for i in range(self.k):
            self.folds_queries.append(pd.read_csv(collection_path + '/fold' + str(i) + '/queries.csv',
                                                  index_col='id_left',
                                                  na_filter=False))
            self.folds_qrels.append(utils.read_qrels(collection_path + '/fold' + str(i) + '/qrels'))
            self.folds_training_qrels.append(utils.read_trec_train_qrels(collection_path + '/fold' + str(i) + '/qrels'))
            
        
    
        
    def update_standard_vocabulary(self,sequences,remove_stopwords=True):
        count = 0
        if remove_stopwords:
            for _,sequence in sequences.iterrows():
                for word in sequence[0].split(" "):
                    temp = word.lower()
                    if temp not in self.stop_words:
                        self.vocabulary[self.stemmer.stem(temp)] += 1
                count +=1
        else:
            for _,sequence in sequences.iterrows():
                for word in sequence[0].split(" "):
                    self.vocabulary[self.stemmer.stem(word.lower())] += 1
                count +=1
        
        
    def build_standard_vocabulary(self,
                                  min_occ = 2,
                                  remove_stopwords = True):
        
        self.vocabulary = Counter()
    
        self.update_standard_vocabulary(self.documents,remove_stopwords)
        
        for i in range(self.k):
            self.update_standard_vocabulary(self.folds_queries[i],remove_stopwords)
                
        del self.vocabulary['']
                
        self.vocabulary = {i+1:elem[0] for i,elem in enumerate(self.vocabulary.most_common()) if elem[1] >= min_occ}

        for key in list(self.vocabulary):
            self.vocabulary[self.vocabulary[key]] = key
            
        self.vocabulary[0] = '<PAD>'
        self.vocabulary['<PAD>'] = 0
    
     
        
    def standard_preprocess(self,
                            remove_stopwords = True,
                            min_occ = 5):
        
        self.build_standard_vocabulary(min_occ = min_occ,
                                       remove_stopwords = remove_stopwords)
                
        self.doc_index,self.indexed_docs = std_tokenizer.index(self.documents,
                                                               self.vocabulary,
                                                               self.stemmer)
        
        self.queries_index = []
        self.indexed_queries = []
        
        for i in range(self.k):
        
            queries_index,indexed_queries = std_tokenizer.index(self.folds_queries[i],
                                                                self.vocabulary,
                                                                self.stemmer)
            self.queries_index.append(queries_index)
            self.indexed_queries.append(indexed_queries)
            
        
        self.all_indexed_queries = []
        for elem in self.indexed_queries:
            self.all_indexed_queries+=elem
        
        self.all_queries_index = dict()
        counter = 0
        for i in range(len(self.queries_index)):
            for j in range(int(len(self.queries_index[i])/2)):
                self.all_queries_index[counter] = self.queries_index[i][j]
                self.all_queries_index[self.queries_index[i][j]] = counter
                counter +=1
    
    
    def build_inverted_index(self):
        
        self.inverted_index = dict()
        

        #Vocabulary object contains both words as strings and their associated integer index
        #Here we add to the inverted index, the integer indexes of words in the vocabulary 
        for token in self.vocabulary:
            if isinstance(token, int):
                self.inverted_index[token] = Counter()
            
        for i,indexed_document in enumerate(self.indexed_docs):
            for token in indexed_document:
                self.inverted_index[token][i] += 1
                
                
    def compute_idf(self):
        nb_docs = len(self.doc_index)
        self.idf = {token:np.log((nb_docs+1)/(1+len(self.inverted_index[token]))) for token in self.inverted_index }
        
        
    def compute_docs_length(self):
        self.docs_length = {i:len(doc) for i,doc in enumerate(self.indexed_docs)}
        
        
    def compute_collection_frequencies(self):
        coll_length = sum([value for key,value in self.docs_length.items()])
        self.c_freq = {token:sum([freq for _,freq in self.inverted_index[token].items()])/coll_length for token in self.inverted_index}
        
    def index_relations(self):
#         self.folds_queries = []
#         self.folds_qrels = []
#         self.folds_training_qrels = []
        
        self.folds_indexed_qrels = []
        self.folds_training_indexed_qrels = []
        
        for i in range(self.k):
        
            training_indexed_qrels = dict()
            training_indexed_qrels['pos'] = []
            training_indexed_qrels['neg'] = dict()
            for elem in self.folds_training_qrels[i]['pos']:
                if elem[1] in self.doc_index:
                    training_indexed_qrels['pos'].append([self.all_queries_index[elem[0]],
                                                          self.doc_index[elem[1]]])

            for key in self.folds_training_qrels[i]['neg']:
                training_indexed_qrels['neg'][key] = []
                for elem in self.folds_training_qrels[i]['neg'][key]:
                    if elem in self.doc_index:
                        training_indexed_qrels['neg'][key].append(self.doc_index[elem])

            self.folds_training_indexed_qrels.append(training_indexed_qrels)

            indexed_qrels = []
            for elem in self.folds_qrels[i]:
                if elem[1] in self.doc_index:
                    indexed_qrels.append([self.all_queries_index[elem[0]],self.doc_index[elem[1]]])
            
            self.folds_indexed_qrels.append(indexed_qrels)
        
        
    def compute_info_retrieval(self):
        self.build_inverted_index()
        self.compute_idf()
        self.compute_docs_length()
        self.compute_collection_frequencies()
        self.index_relations()
        
        
    def save_results(self,index_queries,results,path,top_k=1000):
        with open(path,'w') as f:
            for query,documents in enumerate(results):
                for i,scores in enumerate(documents.most_common(top_k)):
                    f.write(index_queries[query] + ' Q0 ' + self.doc_index[scores[0]] + ' ' + str(i) + ' ' + str(scores[1]) + ' 0\n')
                    
                    
    def pickle_indexed_collection(self,path):
        self.documents = None
        self.folds_queries = None
        with open(path,'wb') as f:
            pickle.dump(self,f)

            
    def compute_fasttext_embedding(self,model_path):
        model = fasttext.load_model(model_path)
        dim = model.get_dimension()
        vocab_size = int(len(self.vocabulary)/2)
        self.embedding_matrix = np.zeros((vocab_size, dim))
        for _ in range(vocab_size):
            self.embedding_matrix[_] = model[self.vocabulary[_]]
                    
            
    def generate_training_batches(self,fold,batch_size=64):
        
        positive_pairs = []
        negative_pairs = {}
        for i in range(self.k):
            if i != fold:
                positive_pairs += self.folds_training_indexed_qrels[i]['pos']
                negative_pairs.update(self.folds_training_indexed_qrels[i]['neg'])
        
        random.shuffle(positive_pairs)
        nb_docs = len(self.indexed_docs)
        nb_train_pairs = len(positive_pairs)
        query_batches = []
        positive_doc_batches = []
        negative_doc_batches = []
        pos = 0
        while(pos + batch_size < nb_train_pairs):
            query_batches.append([q for q,d in positive_pairs[pos:pos+batch_size]])
            positive_doc_batches.append([d for q,d in positive_pairs[pos:pos+batch_size]])
            neg_docs = []
            for elem in query_batches[-1]:
                neg_docs.append(random.choice(negative_pairs[self.all_queries_index[elem]]))
            negative_doc_batches.append(neg_docs)
            pos += batch_size
        return query_batches,positive_doc_batches,negative_doc_batches
    
