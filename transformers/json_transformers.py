import json
from functools import reduce
from nltk import word_tokenize

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