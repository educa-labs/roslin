import json
from functools import reduce
from nltk import word_tokenize
from unidecode import unidecode
from collections import OrderedDict
import pandas as pd

'''
transformer object that turns the data json into arrays of tokenized words.
'''
class JsonTransform():
    
    '''
    add category determines if tag category is added to the final arrays.
    '''
    def __init__(self,add_category=False):
        self.add_category = add_category
        
    '''
    returns arrays of word arrays from tags
    '''    
    def fit(self,X=None,y=None):
        return self
    
    
    def transform(self,X):
        return [ self.process_tag_array(instance) for instance in self.data_to_tags(X)]
    
    
    '''
    returns tag as array of words (no ':').
    add_category determines if the tag category is added to the array.
    '''
    def tag_to_words(self,tag):
        tag = tag.lower()
        category, text = tag.split(":")
        if self.add_category:
            return word_tokenize("{} {}".format(category,text).lower())
        return word_tokenize(text.lower())

    '''
    transforms array of tags into array of words.
    add_category determines if the tag categories are added to the text.
    '''
    def process_tag_array(self,tags):
        aux_tags = list(tags)
        aux_tags[0] = self.tag_to_words(tags[0])
        return reduce(lambda x,y : x + self.tag_to_words(y),aux_tags)

    '''
    returns array of tag arrays from original json.
    '''
    def data_to_tags(self,data):
        return [instance["tags"] for instance in filter(lambda x: x["tags"],data)]

class JsonToTagsTransform(JsonTransform):
    
    def process_tag_array(self,tags):
        return [tag.lower() for tag in tags]

'''
Adds numerical tags to tags transform
'''
class JsonToTagsNumericalTransformer(JsonToTagsTransform):

    def data_to_tags(self,data):
        return [instance["tags"] + JsonToTagsNumericalTransformer.taggize(instance['numerical_tags']) for instance in filter(lambda x: x["tags"],data)]

    @staticmethod
    def taggize(num_tag):
        return ["{}:{}".format(key,value) for key, value in num_tag.items()]

'''
JSON to pandas DataFrame parser used by the KNN model. The resulting data frame
has an equal amount of rows than objects in the JSON file and an equal amount of
columns than the set of tags and numerical tags in the JSON file.
'''
class JSONTransformer:
    def __init__(self):
        self.cols = []

    @staticmethod
    def clear(string):
        string = unidecode(string.lower().strip().replace(' ', '_'))

        return float(string) if string.replace('.', '', 1).isdigit() else string

    def preprocess_json(self,json):
        new_json = []

        for instance in filter(lambda i: i['tags'] is not None and i['numerical_tags'] is not None, json):
            new_instance = OrderedDict()

            for col in self.cols:
                new_instance[col] = None

            # Adding tags. This assumes tags are all categorical values
            for (key, value) in map(lambda t: t.split(':'), instance['tags']):
                key = JSONTransformer.clear(key)

                if new_instance[key] is None:
                    new_instance[key] = JSONTransformer.clear(value)

            # Adding numerical_tags
            for (key, value) in instance['numerical_tags'].items():
                key = JSONTransformer.clear(key)

                if new_instance[key] is None:
                    new_instance[key] = float(value)

            new_json.append(new_instance)

        return new_json

    @staticmethod
    def json_to_df(json):
        df_dict = {
            key: [None] * len(json)
            for instance in json
            for key in instance
        }

        for index, instance in enumerate(json):
            for key, value in instance.items():
                df_dict[key][index] = value

        return pd.DataFrame.from_dict(df_dict)

    '''
    Method necessary to remember the order in which the columns were seen by these
    step of the pipeline. Remember de order is necessary because some null values
    in our data.
    '''
    def fit(self, X=None, y=None):
        for instance in filter(lambda i: i['tags'] is not None and i['numerical_tags'] is not None, X):
            # Adding tags
            for (key, _) in map(lambda t: t.split(':'), instance['tags']):
                key = JSONTransformer.clear(key)
                
                if key not in self.cols:
                    self.cols.append(key)

            # Adding numerical_tags
            for key in instance['numerical_tags']:
                key = JSONTransformer.clear(key)

                if key not in self.cols:
                    self.cols.append(key)

        return self
    
    def transform(self, X=None):
        preprocessed_json = self.preprocess_json(X)
        
        return JSONTransformer.json_to_df(preprocessed_json)
