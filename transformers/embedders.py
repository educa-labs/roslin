import numpy as np
from gensim import corpora
from gensim.models import TfidfModel, LdaModel
from gensim.models.keyedvectors import KeyedVectors


'''
Calculates embedding, mapping a tokenized document to a vector.
To calculate the vector we use a weighted average of precomputed Glove Vectors. 
Weights of the average are given by TFIDF scores.
'''
class TfIdfGloveTransformer():
    
    '''
    word_embedder is pretrained gensim.KeyedVectors model
    
    dim is the dimension on word_embedder
    '''
    def __init__(self,word_embedder,dim=300):
        self.word_embedder = word_embedder
        self.dim=dim
        self.word_dict = None
        self.bows = None
        self.tfidf = None
        self.token2id = None
        
    '''
    Fits from corpus of tokenized documents.
    '''
    def fit(self,X,y=None):
        self.word_dict = corpora.Dictionary(X,prune_at=None)
        self.bows = [self.word_dict.doc2bow(doc) for doc in X]
        self.tfidf = TfidfModel(self.bows,normalize=True)
        self.token2id = self.word_dict.token2id
        return self
    
    
    '''
    returns embedding representation of documents in X
    '''
    def transform(self,X):
        new_bows = [self.word_dict.doc2bow(doc) for doc in X]
        result = np.zeros((len(X),self.dim))
        # perhaps this can be implemented better in a vectorial way
        for i, (doc,bow) in enumerate(zip(X,new_bows)):
            score_hash = { tup[0]:tup[1] for tup in self.tfidf.__getitem__(bow,-1)} # threshold
            weighted_embeddings = np.array([np.dot(self.word_embedder[word],score_hash[self.token2id[word]]) if word in self.word_embedder else np.zeros((1,self.dim)) for word in doc])
            result[i] = np.sum(weighted_embeddings, axis=0)
        return result
            
"""
Generates doc embeddings baed on topic modelling.
Does Tf-Idf transformation and then computes probability distibutions with LDA algorithm.
"""
class LdaTransformer():
    """
    dim: amount of topics to model. aka output vector dimension.
    """
    def __init__(self,dim=20):
        self.dim=dim
        self.word_dict = None
        self.bows = None
        self.tfidf = None
        self.token2id = None
        self.lda = None
    
    def fit(self,X,y=None):
        self.word_dict = corpora.Dictionary(X,prune_at=None)
        self.bows = [self.word_dict.doc2bow(doc) for doc in X]
        self.tfidf = TfidfModel(self.bows,normalize=True)
        self.token2id = self.word_dict.token2id
        self.lda = LdaModel(self.tfidf[self.bows],num_topics=self.dim,minimum_probability=0)
        return self
    
    """
    receives tokenized documents and returns the distribution of each.
    """
    def transform(self,X):
        new_bows = [self.word_dict.doc2bow(doc) for doc in X]
        distributions = np.array(self.lda[self.tfidf[new_bows]])
        return np.reshape(np.delete(distributions,np.s_[:1],2),(len(X),self.dim))