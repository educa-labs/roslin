from sklearn.neighbors import BallTree

'''
wrapper for sklearn BallTree that can be added to a pipeline
'''

class BallTreePredictor():
    
    def __init__(self,k=5):
        self.tree = None
        self.k=k
        
    def set_neighbors(self,k):
        self.k = k
        
    def fit(self,X,y=None):
        self.tree = BallTree(X)
        return self
        
    def predict(self,X):
        return self.tree.query(X,self.k)