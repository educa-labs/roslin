import json as js
import pandas as pd


# Hielo
from transformers.embedders import LdaTransformer
from transformers.json_transformers import JsonToTagsTransform


# Dimensionality reduction algorithms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def reduce_dimensionality(raw_json, algorithm=TSNE):
    json_transformer = JsonToTagsTransform()
    documents = json_transformer.transform(raw_json)
    lda = LdaTransformer().fit(documents)

    X = lda.transform(documents)

    return algorithm(n_components=2).fit_transform(X)


def formatter(X_embedded, raw_json):
    def formatter(x):
        embedded, raw = x
        _id = raw['id']

        return (*embedded, _id, 'images/{0}'.format(_id))

    data_frame = pd.DataFrame(list(map(formatter, zip(X_embedded, raw_json))))
    header = ['0', '1', 'id', 'file']

    data_frame.to_csv('output.csv', header=header, index=False)


if __name__ == '__main__':
    raw_json = js.load(open('data.json', encoding='utf-8'))

    formatter(reduce_dimensionality(raw_json), raw_json)
