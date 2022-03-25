import sys
import argparse
import pandas as pd
import numpy as np
from util.embeddings import SemanticEmbedding, EmbeddingCache


def get_similarity_matrix(words, model_path, **kwargs):
    if len(words) == 2:
        return SemanticEmbedding.get_similarity_matrix(words, model_path)[0, 0]
    return SemanticEmbedding.get_similarity_matrix(words, model_path)


def cache_embeddings(words, model_path, **kwargs):
    """
    Example args: `cache_embeddings ^leader\W$ ^trustworthy$`
    """
    return EmbeddingCache.cache_embeddings(words, model_path)


def imagenet_similarity_matrix(parent, id_word_path, parent_child_path, keywords, model_path, n=10, **kwargs):
    # get the similarity matrix
    id_word = pd.read_csv(id_word_path, sep="\t", header=None)
    id_word.columns = ["id", "token"]
    parent_child = pd.read_csv(parent_child_path, sep=" ", header=None)
    parent_child.columns = ["parent", "child"]
    print("Finding children for parent {}".format(parent))
    tokens = pd.merge(
        id_word, parent_child[parent_child["parent"] == parent],
        how="right", left_on="id", right_on="child"
    )[["id", "token"]]
    print(tokens)
    sim_matrix = get_similarity_matrix(
        keywords + _parse_token_list_to_regex(tokens["token"].values),
        model_path
    )

    # find the min/max for each keyword
    np.fill_diagonal(sim_matrix, 0)
    for i in range(len(keywords)):
        print("## For keyword {} ##".format(keywords[i]))

        # find the indices of the five largest similarity scores in this keyword's row
        # of the similarity matrix
        print("- Top Five Trustworthy Terms -")
        print(tokens["token"][_get_min_max_n_indices(sim_matrix[:, i], n, top_n=True)])
        print("- Bottom Five Trustworthy Terms -")
        print(tokens["token"][_get_min_max_n_indices(sim_matrix[:, i], n, top_n=False)])

        print("- Bottom Five Most Trustworthy Terms -")

    return sim_matrix


def _get_min_max_n_indices(a, n, top_n=True):
    i = -n if top_n else n
    partition = np.argpartition(a, i)
    return partition[i:] if top_n else partition[:i]


def _parse_token_list_to_regex(tokens):
    return ["^{}$".format(x) for x in tokens]


# def file_to_dict(path, sep=" "):
#     d = {}
#     with open(path) as f:
#         for line in f:
#             (key, val) = line.strip("\n").split(sep)
#             d[key] = val
#     return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the similarity of two words using semantic embeddings.")
    parser.add_argument(
        'endpoint',
        type=str,
        help="the endpoint to run"
    )
    endpoint = sys.argv[1]
    method = None

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/glove.840B.300d.txt',
        help="Path to GloVe mapping to use"
    )
    if endpoint in ["cache_embeddings", "get_similarity"]:
        parser.add_argument(
            'words',
            nargs='+',
            type=str,
            help="Path to list of words to save embeddings for"
        )
    print(endpoint)
    if endpoint == "imagenet_similarity_matrix":
        parser.add_argument(
            'parent',
            type=str,
            help="Synset ID of parent to gather child IDs for"
        )
        parser.add_argument(
            'id_word_path',
            type=str,
            help="Path to Synset ID / word table"
        )
        parser.add_argument(
            'parent_child_path',
            type=str,
            help="Path to parent ID / child ID table"
        )
        parser.add_argument(
            '--keywords',
            nargs='+',
            type=str,
            help="Keyword regex tokens to find min/max similarity for - e.g. `^trustworthy$`"
        )
        method = imagenet_similarity_matrix
    elif endpoint == "cache_embeddings":
        method = cache_embeddings
    elif endpoint == "get_similarity":
        method = get_similarity_matrix
    else:
        raise ValueError("invalid endpoint")
    args = parser.parse_args()
    print(method(**vars(args)))
