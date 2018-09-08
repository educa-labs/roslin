from pymongo import MongoClient
from sklearn.neighbors import BallTree, DistanceMetric
import pickle


class MongoSerializable:
    def save(self, db_name, key):
        with MongoClient() as client:
            db = client[db_name]

            _object      = {}
            _object[key] = pickle.dumps(self)

            db.models.insert(_object)

'''
wrapper for sklearn BallTree that can be added to a pipeline
'''

class BallTreePredictor(MongoSerializable):
    
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


def object_cols_to_category(data_frame):
    # Set <object> dtype columns to <category>

    cat_cols = data_frame.select_dtypes(include=['object']).columns.values
    
    for col in cat_cols:
        data_frame[col] = data_frame[col].astype('category')

    return data_frame


def category_cols_to_codes(data_frame, cat_cols, cat_cols_codes=None):
    # Set <category> dtype columns values to <int8>

    if cat_cols_codes is None:
        cat_cols_codes = {}
    
        for col in cat_cols:
            cat_code_mapping = {
                cat: code
                for cat, code in zip(data_frame[col], data_frame[col].cat.codes)
            }

            cat_cols_codes[col] = cat_code_mapping
    
    for col in cat_cols:
        new_col = []
        
        for value in data_frame[col]:
            new_col.append(cat_cols_codes[col][value] if value else 0)
        
        data_frame[col] = new_col

    return data_frame, cat_cols_codes


class GowerDistance:
    def __init__(self, cols_hash, cat_cols, con_cols, W_i, R_i):
        self.cols_hash = cols_hash
        self.cat_cols  = cat_cols
        self.con_cols  = con_cols
        self.W_i       = W_i
        self.R_i       = R_i
        self.W_i_sum   = np.sum(W_i)  # Micro-optimization
    
    @staticmethod
    def cat_dist(c_j, c_k):
        # Categorical distance function

        return int(not c_j == c_k)

    @staticmethod
    def con_dist(x_j, x_k, r_i):
        # Continuous distance function

        return 1 - np.divide(np.absolute(x_j - x_k), r_i)
    
    def __call__(self, X_j, X_k):
        distance = 0

        for col in self.cat_cols:
            distance += np.dot(self.W_i[self.cols_hash[col]], GowerDistance.cat_dist(X_j[self.cols_hash[col]], X_k[self.cols_hash[col]]))

        for col in self.con_cols:
            distance += np.dot(self.W_i[self.cols_hash[col]], GowerDistance.con_dist(X_j[self.cols_hash[col]], X_k[self.cols_hash[col]], self.R_i[self.cols_hash[col]]))

        return distance / self.W_i_sum


class KNNPredictor(MongoSerializable):
    def __init__(self, k=5):
        self.k = k
        
        self.cat_cols       = None
        self.cat_cols_codes = None
        self.con_cols       = None
        
        self.df   = None
        self.tree = None

    def fit(self, X, y=None):
        df = object_cols_to_category(X)

        self.cat_cols = df.select_dtypes(include=['category']).columns.values
        self.con_cols = df.select_dtypes(include=['float64']).columns.values

        self.df, self.cat_cols_codes = category_cols_to_codes(df, self.cat_cols)

        cols_hash = { col: i for i, col in enumerate(df.columns.values) }

        W_i = [.66 if col in self.cat_cols else 1 for col in cols_hash]
        R_i = [np.max(df[col]) - np.min(df[col]) if col in self.con_cols else 1 for col in cols_hash]

        gower_distance = GowerDistance(cols_hash, self.cat_cols, self.con_cols, W_i, R_i)
        metric         = DistanceMetric.get_metric('pyfunc', func=gower_distance)
        self.tree      = BallTree(df, metric=metric)
        
        return self
    
    def predict(self, X):
        X    = object_cols_to_category(X)
        X, _ = category_cols_to_codes(X, self.cat_cols, cat_cols_codes=self.cat_cols_codes)

        prediction = self.tree.query(X, self.k, return_distance=False)
        
        return list(map(lambda p: self.df.iloc[p], prediction))
