import json
import numpy as np


# Encoder to save numpy arrays
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super.default(self, obj)


#returns documents with id_key in ids
def get_by_id(data,ids,id_key='quick_code'):
    if type(ids) == list:
        return list(filter(lambda x: x[id_key] in ids, data))

# returns array mapping data_index to data[id_key]
def map_to_id(data,id_key='quick_code'):
    return [document[id_key] for document in data]

# returns hash mapping  data[id_key] to data_index 
def map_from_id(data,id_key='quick_code'):
    return { document[id_key]:i for i,document in enumerate(data)}

# same as get_by_id but uses the mapping to improve performance. 
def get_by_id_map(data,ids,map):
    return [ data[map[id]] for id in ids]