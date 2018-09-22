import os
import pickle


# Mongo
from pymongo import MongoClient


MONGODB_URI_KEY = 'MONGODB_URI'


def save(model, db_name, collection_name, key, client):
    db = client[db_name]
    collection = db[collection_name]

    _object = {}
    _object[key] = pickle.dumps(model)

    collection.insert(_object)


def build_filter(key):
    _filter = {}
    _filter[key] = {'$exists': True}

    return _filter


def load_model(collection, key):
    print('Finding model <{0}>'.format(key.upper()))

    _filter = build_filter(key)
    count = collection.count_documents(_filter)

    print('Found {0} <{1}> models...'.format(count, key.upper()))

    if count > 0:
        print('Picking the last one...')

        cursor = collection.find(_filter).sort([('_id', -1)]).limit(1)

        return pickle.loads(cursor[0][key])


def load_models(db_name, collection_name, models):
    loaded_models = {}

    if MONGODB_URI_KEY in os.environ:
        uri = os.environ[MONGODB_URI_KEY]

        print('Found MONGODB_URI environment variable: {0}'.format(uri))

        client = MongoClient(uri)
    else:
        client = MongoClient()

    with client:
        db = client[db_name]
        collection = db[collection_name]

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
