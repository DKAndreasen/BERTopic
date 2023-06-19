import numpy as np
import pandas as pd
import plotly.graph_objects as go

from umap import UMAP
from typing import List, Union


def visualize_splitted_documents(topic_model,
                                 docs: List[str],
                                 original_doc_ref: List[int],
                                 hide_unassigned_snippets_from_assigned_docs: bool = True,
                                 topics: List[int] = None,
                                 embeddings: np.ndarray = None,
                                 reduced_embeddings: np.ndarray = None,
                                 sample: float = None,
                                 hide_annotations: bool = False,
                                 hide_document_hover: bool = False,
                                 custom_labels: Union[bool, str] = False,
                                 title: str = "<b>Documents and Topics</b>",
                                 doc_string_joiner: str = 'first',
                                 width: int = 1200,
                                 height: int = 750):
    """ Visualize documents and their topics in 2D

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        original_doc_ref: A list indicating which docs comes from the same original document.
                           That is, which docs should, within each topic, be merged to one doc.
        hide_unassigned_snippets_from_assigned_docs: Hide unassigned document snippets
            from original documents where other snippets have been assigned to one or
            more topics (alternatively these snippets will all be visualised individually
            instead of being merged)
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualization.
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        doc_string_joiner: string that the docs are joined upon with the reduced embedding
                           coordinates are averaged, eg. ' /--/ '. In case the embeddings
                           are provided seperately and docs are used to set a hover text, then
                           the special str 'first' (default) can be used to indicate that only
                           the first doc entry for a group of docs assigned to the same topic
                           from the same original document, should be used.
        width: The width of the figure.
        height: The height of the figure.

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_splitted_documents(docs,original_docs_ref)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and prefered pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Split documents
    raw_docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    seq_len = sentence_model.max_seq_length - len(sentence_model.tokenizer.encode(''))
    def doc_splitter(doc:str) -> List[str]:
        encoded_sent = sentence_model.tokenizer.encode(doc,add_special_tokens=False)
        return [sentence_model.tokenizer.decode(
                encoded_sent[i*seq_len:(i+1)*seq_len]
            ) for i in range(np.ceil(len(encoded_sent)/seq_len).astype(int))]
    splitted_docs = pd.Series([doc_splitter(doc) for doc in docs]).explode()
    original_docs_ref = splitted_docs.index
    docs = list(splitted_docs)

    # Prepare embeddings
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_splitted_documents(docs, original_docs_ref, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_splitted_documents(docs, original_docs_ref, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_splitted_documents(docs, original_docs_ref, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    <iframe src="../../getting_started/visualization/documents.html"
    style="width:1000px; height: 800px; border: 0px;""></iframe>
    """
    topic_per_doc = topic_model.topics_

    # Assuming many documents have been split to fit the token requirement of the sentence transformer, we want to
    # reduce the number of snippets representing the same original document within the same topic class, while allowing
    # a document to contain multiple topics
    _, split_groups = np.unique(list(zip(topic_per_doc, original_doc_ref)), axis=0, return_inverse=True)

    df_full = pd.DataFrame(
        {
            "topic": np.array(topic_per_doc),
            "doc": docs,
            "orig_doc": original_doc_ref,
            "split_group": split_groups
        }
    )
    if hide_unassigned_snippets_from_assigned_docs:
        # If one or more snippets of an original document is already assigned to a topic, we will not
        # display snippets from that document, that were not assigned a to a topic class (alternatively we will not merge
        # unassigned snippet representations
        df_full.drop(
            df_full[(df_full.topic == -1)].index[
                df_full[(df_full.topic == -1)]['orig_doc'].isin(
                    df_full['orig_doc'][(df_full.topic != -1)].unique() # list of original docs assigned to some topic
                )
            ],
            inplace=True
        )

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    # We ensure that we cannot sample multiple time from the same original document
    df_sample = df_full.groupby('topic',group_keys=False).apply(
        lambda topic_df: topic_df.drop_duplicates(
            subset=['split_group']
        ).sample(
            n=int(len(topic_df.drop_duplicates(subset=['split_group']))*sample),
            replace=False
        ) if len(topic_df.drop_duplicates(subset=['split_group'])) >= 100 else topic_df
    )

    df = df_sample[df_sample["topic"] == -1].append(
        df_full[(df_full['topic'] != -1) & df_full.split_group.isin(df_sample['split_group'])]
    ).sort_index()

    # # Sample the data to optimize for visualization and dimensionality reduction
    # if sample is None or sample > 1:
    #     sample = 1
    #
    # indices = []
    # for topic in set(topic_per_doc):
    #     s = np.where(np.array(topic_per_doc) == topic)[0]
    #     size = len(s) if len(s) < 100 else int(len(s) * sample)
    #     indices.extend(np.random.choice(s, size=size, replace=False))
    # indices = np.array(indices)
    #
    # df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    # df["doc"] = [docs[index] for index in indices]
    # df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[df.index]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[df.index]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Average classified document snippets by a simple mean (the mean could be weigted by the classification
    # probability)
    df = df[df.topic == -1].drop(columns='split_group').append(
        df[df.topic != -1].groupby('split_group').agg({
            'topic': "first",
            'doc': doc_string_joiner if doc_string_joiner == "first" else lambda x: doc_string_joiner.join(x),
            'orig_doc': "first",
            'x': "mean",
            'y': "mean"
        }),
        ignore_index=True
    )

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        names = [f"{topic}_" + "_".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in unique_topics]

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
