import json
import numpy as np


# Encoder to save numpy arrays
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super.default(self, obj)


#returns documents with id in ids
def get_by_id(data,ids,id_key='quick_code'):
    if type(ids) == list:
        return list(filter(lambda x: x[id_key] in ids, data))

def map_to_id(data,id_key='quick_code'):
    return [document[id_key] for document in data]

def map_from_id(data,id_key='quick_code'):
    return { document[id_key]:i for i,document in enumerate(data)}


def get_by_id_map(data,ids,map):
    return [ data[map[id]] for id in ids]