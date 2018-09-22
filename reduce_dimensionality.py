import json as js
import pandas as pd

from sklearn.pipeline import Pipeline

# Dimensionality reduction algorithms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Pipeline builders
from builders import KNN_BUILDER
from builders import LDA_TAGS_AVERAGE_BUILDER, LDA_TAGS_GREEDY_BUILDER, LDA_WORDS_AVERAGE_BUILDER, LDA_WORDS_GREEDY_BUILDER
from builders import GLOVE_AVERAGE_BUILDER, GLOVE_GREEDY_BUILDER


def reduce_dimensionality(pipeline, raw_json, algorithm=TSNE):
    json_transformer = pipeline.steps[0][1]
    documents = json_transformer.transform(raw_json)
    lda = pipeline.steps[1][1].fit(documents)

    X = lda.transform(documents)

    return algorithm(n_components=2).fit_transform(X)


def formatter(X_embedded, raw_json, output_filename):
    def formatter(x):
        embedded, raw = x
        _id = raw['id']

        return (*embedded, _id, 'images/{0}'.format(_id))

    data_frame = pd.DataFrame(list(map(formatter, zip(X_embedded, raw_json))))
    header = ['0', '1', 'id', 'file']

    data_frame.to_csv(output_filename, header=header, index=False)


if __name__ == '__main__':
    raw_json = js.load(open('data.json', encoding='utf-8'))

    pipelines = [
        (LDA_TAGS_AVERAGE_BUILDER(), 'visualization/lda_av.csv'),
        (LDA_TAGS_GREEDY_BUILDER(), 'visualization/lda_greedy.csv'),
        (LDA_WORDS_AVERAGE_BUILDER(), 'visualization/ldaw_av.csv'),
        (LDA_WORDS_GREEDY_BUILDER(), 'visualization/ldaw_greedy.csv'),
        (GLOVE_AVERAGE_BUILDER(), 'visualization/glove_av.csv'),
        (GLOVE_GREEDY_BUILDER(), 'visualization/glove_greedy.csv'),
    ]

    for pipeline, output_filename in pipelines:
        formatter(
            X_embedded=reduce_dimensionality(pipeline, raw_json),
            raw_json=raw_json,
            output_filename=output_filename
        )
