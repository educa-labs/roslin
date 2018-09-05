from sklearn import neighbors
import json
import numpy as np
import pandas as pd
import unidecode


def categorical_distance(c_j, c_k):
  return int(not c_j == c_k)


def continuous_distance(x_j, x_k, r_i):
  return np.divide(np.absolute(x_j - x_k), r_i)


def gower_distance(X_j, X_k, categorical_columns, continuous_columns):
  # column_hash
  # W_i

  distance = 0

  for col in categorical_columns:
    distance += np.dot(W_i[column_hash[col]], categorical_distance(X_j[column_hash[col]], X_k[column_hash[col]]))

  for col in continuous_columns:
    distance += np.dot(W_i[column_hash[col]], continuous_distance(X_j[column_hash[col]], X_k[column_hash[col]], R_i[column_hash[col]]))

  distance += W_i[11] * (1 - np.absolute(similarity_matrix[int(X_j[column_hash['ID']]), int(X_k[column_hash['ID']])]))
  
  return distance


class JSONParser:
  def __init__(self, path):
    self.raw_json = json.load(open(path, encoding='utf-8'))
    self.json = JSONParser.preprocess_json(self.raw_json)
    self.table = JSONParser.json_to_table(self.json)

  @staticmethod
  def clear(string):
    string = unidecode.unidecode(string.lower().strip().replace(' ', '_'))

    return float(string) if string.replace('.', '', 1).isdigit() else string

  @staticmethod
  def preprocess_json(json):
    new_json = []

    for instance in json:    
      new_instance = {}

      for (key, value) in map(lambda t: t.split(':'), instance['tags']):
        key = JSONParser.clear(key)

        if key not in new_instance:
          new_instance[key] = JSONParser.clear(value)

      new_json.append(new_instance)
    
    return new_json

  @staticmethod
  def json_to_table(json):
    data = {
      key: [None] * len(json)
      for instance in json
      for key in instance
    } 

    for index, instance in enumerate(json):
      for key, value in instance.items():
        data[key][index] = value
    
    return pd.DataFrame.from_dict(data)


if __name__ == '__main__':
  parser = JSONParser('./../data.json')

  categorical_columns = parser.table.select_dtypes(include=[object]).columns.values
  continuous_columns = parser.table.select_dtypes(include=[np.float]).columns.values

  print(categorical_columns, continuous_columns)

  metric = neighbors.DistanceMetric.get_metric('pyfunc', func=gower_distance, { categorical_columns: categorical_columns })

  # tree = neighbors.BallTree(parser.table, metric=metric)
