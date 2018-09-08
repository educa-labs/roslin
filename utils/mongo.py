from pymongo import MongoClient
import pickle


def save_model(model, db, key):
    obj      = {}
    obj[key] = pickle.dumps(model)

    db.models.insert(obj)


def open_db(db_name):
    client = MongoClient()

    return client[db_name]
