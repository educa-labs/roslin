from sklearn.neighbors import BallTree, DistanceMetric
from utils.json_utils import map_to_id, map_from_id

import json as js
import numpy as np
import pandas as pd
import pickle
import math


'''
wrapper for sklearn BallTree that can be added to a pipeline
'''
class BallTreePredictor:
    def __init__(self, k=5, average=False):
        self.tree = None
        self.k = k
        self.average = average

    def set_k(self, k):
        self.k = k

    def fit(self, X, y=None):
        self.tree = BallTree(X)
        return self

    def transform(self, X):
        if not self.average:
            return self.tree.query(X, self.k)
        else:
            return self.tree.query(np.array([np.mean(X, axis=0)]), self.k)


'''
Given a panda's DataFrame object, it returns the same given DataFrame but whose
object type columns were changed to category type. This is useful because the
API that pandas offer us in categorical type columns.
'''
def object_cols_to_category(data_frame):
    # Set <object> dtype columns to <category>

    cat_cols = data_frame.select_dtypes(include=['object']).columns.values

    for col in cat_cols:
        data_frame[col] = data_frame[col].astype('category')

    return data_frame

'''
Given a panda's DataFrame object, it return the same given DataFrame but whose
category type columns values were replaced by numerical values, using the
pandas API "data_frame[col].cat.codes".
'''
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
        # Checking null values.
        if col not in data_frame:
            new_col = [-1] * data_frame.shape[0]
        else:
            new_col = []

            for value in data_frame[col]:
                #  Checking null values.
                if type(value) == float and math.isnan(value):
                    new_col.append(-1)
                else:
                    new_col.append(cat_cols_codes[col][value] )

        data_frame[col] = new_col

    return data_frame, cat_cols_codes

'''
We are placing vector with categorical data in a continuos multi-dimensional
vectorial space assignin a number to each one of the posibile values of those
categories. To avoid give more weight to a given category value just by the
order of how their values were given, we need a new distance metric.

We decided to use Gower Distance (https://stats.stackexchange.com/questions/1731
44/convert-categorical-data-to-numerical-data-to-compute-a-distance-then), which
in fact allow us to decide arbitrari weight for each column. As default value we
decided to weight by .66 all the categorical columns and by 1 all the numerical
columns. Feel free to edit COLUMNS_WEIGHTS values as you see fit.
'''

COLUMNS_WEIGHTS = {
    'accesorios': .66,
    'color_cubierta': .66,
    'configuracion': .66,
    'contraste': .66,
    'cubierta': .66,
    'diseno_y_creatividad': 1,
    'ejecucion': 1,
    'espacialidad': .66,
    'espesor_cubierta': .66,
    'estilo': .66,
    'exigencias_tecnicas': 1,
    'lineas': .66,
    'luminosidad': .66,
    'materialidad': .66,
    'modulos': .66,
    'percepcion_de_tamano': .66,
    'precio_estimado': 1,
    'proporcion_cajones_a_puertas': 1,
    'textura': .66,
    'tonalidad': .66,
    'valor_casa': 1,
    'visualizacion': .66,
    'volumetrias': .66,
}

class GowerDistance:
    def __init__(self, cols_hash, cat_cols, con_cols, W_i, R_i):
        self.cols_hash = cols_hash
        self.cat_cols = cat_cols
        self.con_cols = con_cols
        self.W_i = W_i
        self.R_i = R_i
        self.W_i_sum = np.sum(list(W_i.values()))  # Micro-optimization

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
            distance += np.dot(self.W_i[col], GowerDistance.cat_dist(
                X_j[self.cols_hash[col]], X_k[self.cols_hash[col]]))

        for col in self.con_cols:
            distance += np.dot(self.W_i[col], GowerDistance.con_dist(
                X_j[self.cols_hash[col]], X_k[self.cols_hash[col]], self.R_i[col]))

        return distance / self.W_i_sum


'''
Wrapper for sklearn BallTree that can be added to a pipeline, but with a higher
amout of logic. Basicly it receives to pandas DataFrame, it preprocess this
data frame to obtain a new data frame with only numerical values, creates the
metric distance and gives you necessary parameters to the BallTree predictor.
'''
class KNNPredictor:
    def __init__(self, k=5):
        self.k = k

        self.cat_cols = None
        self.cat_cols_codes = None
        self.con_cols = None

        self.df = None
        self.tree = None

    def fit(self, X, y=None):
        df = object_cols_to_category(pd.DataFrame(X))

        self.cat_cols = df.select_dtypes(include=['category']).columns.values
        self.con_cols = df.select_dtypes(include=['float64']).columns.values

        self.df, self.cat_cols_codes = category_cols_to_codes(
            df, self.cat_cols)

        cols_hash = {col: i for i, col in enumerate(self.df.columns.values)}

        W_i = COLUMNS_WEIGHTS
        R_i = {col: np.max(self.df[col]) - np.min(self.df[col]) for col in self.con_cols}
        gower_distance = GowerDistance(
            cols_hash, self.cat_cols, self.con_cols, W_i, R_i)
        metric = DistanceMetric.get_metric('pyfunc', func=gower_distance)
        self.tree = BallTree(self.df, metric=metric)

        return self

    def transform(self, X):
        X = object_cols_to_category(X)
        X, _ = category_cols_to_codes(
            data_frame=X,
            cat_cols=self.cat_cols,
            cat_cols_codes=self.cat_cols_codes
        )
        prediction = self.tree.query(X, self.k)

        return prediction

    def set_k(self, k):
        self.k = k
