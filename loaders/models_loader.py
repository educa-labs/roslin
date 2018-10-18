from collections import OrderedDict
from pymongo import MongoClient

import os
import pickle


# Key used to find the MongoDB uri in the env variables of the machine. If it      
# exists, we are at a remote machine.
MONGODB_URI_KEY = 'MONGODB_URI'

'''
Dumps the model and inserts it in the specified collection of the specified
database using <key> as document key.
'''
def save(model, db_name, collection_name, key, client):
    db = client[db_name]
    collection = db[collection_name]

    _object = {}
    _object[key] = pickle.dumps(model)

    collection.insert(_object)

'''
Creates a filter to find for documents of key <key>.
'''
def build_filter(key):
    _filter = {}
    _filter[key] = {'$exists': True}

    return _filter

'''
Search into the collection for all the documents of key <key>. Since the documents
are basically serialized models, it search for the last serialized model under this
key.
'''
def load_model(collection, key):
    print('Finding model <{0}>'.format(key.upper()))

    _filter = build_filter(key)
    count = collection.count_documents(_filter)

    print('Found {0} <{1}> models...'.format(count, key.upper()))

    if count > 0:
        print('Picking the last one...')

        cursor = collection.find(_filter).sort([('_id', -1)]).limit(1)

        return pickle.loads(cursor[0][key])


'''
Connects to the database (local or remote) to find all models specified in <models>
keys. If it doesn't find a given model, it trains and saves a new one. If <clear> is
True, it will clear the specified collection before all the process, so he will
train all the models from scratch.
'''
def load_models(collection_name, models, clear=False):
    loaded_models = OrderedDict()

    if MONGODB_URI_KEY in os.environ:
        uri = os.environ[MONGODB_URI_KEY]
        db_name = os.environ['MONGO_DB_DB']
        print('Found MONGODB_URI environment variable: {0}'.format(uri))

        client = MongoClient(uri)
    else:
        db_name = 'test_database'
        client = MongoClient()

    with client:
        db = client[db_name]
        collection = db[collection_name]

        if clear:
            print('Cleaning models collection...')

            collection.remove({})

        for key in models:
            model = load_model(collection, key)

            if model is None:
                print('Training model <{0}>'.format(key.upper()))

                builder, raw_json = models[key]
                model = builder()

                model.fit(raw_json)

                save(
                    model=model,
                    db_name=db_name,
                    collection_name=collection_name,
                    key=key,
                    client=client
                )

            loaded_models[key] = model

    return loaded_models
