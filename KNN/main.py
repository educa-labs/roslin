import json
import pandas as pd


class JSONParser:
  def __init__(self, path):
    self.raw_json = json.load(open(path, encoding='utf-8'))
    self.json = JSONParser.preprocess_json(self.raw_json)
    self.table = JSONParser.json_to_table(self.json)

  @staticmethod
  def preprocess_json(json):
    new_json = []

    for instance in json:    
      new_instance = {}

      for (key, value) in map(lambda t: t.split(':'), instance['tags']):
        # Solo añade el primer key que encuentra
        if key not in new_instance:
          new_instance[key] = value

      new_json.append(new_instance)
    
    return new_json

  @staticmethod
  def json_to_table(json):
    data = {
      key.lower(): [None] * len(json)
      for instance in json
      for key in instance
    } 

    for index, instance in enumerate(json):
      for key, value in instance.items():
        data[key.lower()][index] = value
    
    return pd.DataFrame.from_dict(data)


if __name__ == '__main__':
  parser = JSONParser('./../data.json')
