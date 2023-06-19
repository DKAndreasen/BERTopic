import copy
import pytest
import numpy as np
from umap import UMAP
from bertopic import BERTopic

@pytest.fixture()
def splitted_documents(documents, embedding_model):
    max_seq_length = 128
    seq_len = max_seq_length - len(embedding_model.tokenizer.encode(''))

    def doc_splitter(doc):
        encoded_sent = embedding_model.tokenizer.encode(doc, add_special_tokens=False)
        return [embedding_model.tokenizer.decode(
            encoded_sent[i * seq_len:(i + 1) * seq_len]
        ) for i in range(np.ceil(len(encoded_sent) / seq_len).astype(int))]

    splitted_docs = [memb for i, doc in zip(range(len(documents)), documents) for split in doc_splitter(doc) for memb in [i, split]]
    original_doc_ref = splitted_docs[::2]
    docs = splitted_docs[1::2]
    return docs, original_doc_ref


@pytest.fixture()
def split_embeddings(splitted_documents, embedding_model):
    splits = splitted_documents[0]
    embeddings = embedding_model.encode(splits)
    return embeddings


@pytest.fixture()
def reduced_split_embeddings(split_embeddings):
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(split_embeddings)
    return reduced_embeddings

@pytest.fixture()
def split_topic_model(splitted_documents, split_embeddings, embedding_model):
    splits = splitted_documents[0]
    model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True)
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 3
    model.fit(splits, split_embeddings)
    return model


def test_splitted_documents(split_topic_model, reduced_split_embeddings, splitted_documents):
    topic_model = split_topic_model
    splits, original_doc_ref = splitted_documents
    topics = set(topic_model.topics_)
    fig = topic_model.visualize_splitted_documents(splits, original_doc_ref, embeddings=reduced_split_embeddings, hide_document_hover=True)
    fig_topics = [int(data["name"].split("_")[0]) if (data["name"].split("_")[0] != 'other') else -1 for data in fig.to_dict()["data"]]
    assert set(fig_topics) == topics
