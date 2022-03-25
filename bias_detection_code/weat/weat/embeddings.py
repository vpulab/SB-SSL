import pandas as pd
import pandas.errors
from sklearn.metrics.pairwise import cosine_similarity
import os


class SemanticEmbedding:
    def __init__(self, token, model_path):
        self.token = token
        self.model_path = model_path
        self.embedding = self.get_embedding()[0]

    def get_embedding(self):
        return EmbeddingCache.query_cache([self.token], self.model_path)[:, 1:]

    @staticmethod
    def get_similarity_matrix(words, model_path):
        # cache first to save time later
        cached_words = EmbeddingCache.cache_embeddings(words, model_path)
        print(SemanticEmbedding(cached_words[0], model_path).embedding)
        return cosine_similarity([SemanticEmbedding(word, model_path).embedding for word in cached_words])


class EmbeddingCache:
    cache_path = '.cache/cache.csv'
    cache = None

    @staticmethod
    def get_cache():
        if EmbeddingCache.cache is None:
            EmbeddingCache.cache = EmbeddingCache.load_embeddings(EmbeddingCache.cache_path)
        return EmbeddingCache.cache

    @staticmethod
    def set_cache(cache):
        EmbeddingCache.cache = cache

    @staticmethod
    def cache_embeddings(words, model_path):
        """
        Finds and caches new embeddings for easy access

        :param words: a list of regex tokens to match semantic embeddings to be added to the cache
        :param model_path: the path of the GloVe model .txt file to use
        :return: those words that were successfully cached
        """
        if len(words) < 1:
            return
        cache = EmbeddingCache.get_cache()
        embeddings = EmbeddingCache.load_embeddings(model_path, cache_embeddings=True)
        targets = EmbeddingCache.regex_df_column(words, embeddings, 'keys').copy()
        print(cache.shape if cache is not None else "None")
        if cache is None:
            cache = targets
        else:
            cache.update(targets)
        print(cache.shape)
        EmbeddingCache.dump_embeddings(cache, EmbeddingCache.cache_path)
        cached_words = targets['keys'].values
        if len(cached_words) != len(words):
            missing_words = set(words) - set(cached_words)
            print("Warning: no matches in model for {} words:\n {}".format(len(missing_words), missing_words))
        return targets['keys'].values

    @staticmethod
    def regex_df_column(expressions, df, column):
        if len(expressions) < 1:
            return None
        if len(expressions) == 1:
            return df[df[column].str.contains(expressions[0])]
        return df[df[column].str.contains("|".join(expressions))]

    @staticmethod
    def load_embeddings(path, cache_embeddings=False):
        print("Loading embeddings from {}".format(path))
        try:
            if cache_embeddings:
                try:
                    df = pd.read_pickle(EmbeddingCache._get_pickle_path(path))
                    return df
                except (FileNotFoundError, EOFError):
                    pass
            df = pd.read_csv(path, sep=" ", header=None)
            df.rename(columns={0: 'keys'}, inplace=True)
            df['keys'] = df['keys'].str.lower().astype(str)
            if cache_embeddings:
                df.to_pickle(EmbeddingCache._get_pickle_path(path))
            return df
        except (pandas.errors.EmptyDataError, FileNotFoundError):
            print("No cache data found.")
            return None

    @staticmethod
    def dump_embeddings(df, path):
        with open(path, 'w+') as f:
            df.to_csv(f, sep=" ", header=False, index=False, line_terminator='\n')
            f.close()
        print("Dumped embeddings to {}".format(path))

    @staticmethod
    def query_cache(words, model_path):
        cache = EmbeddingCache.get_cache()
        if cache is None:
            print("No cache. Generating new cache...")
            cache = EmbeddingCache.cache_embeddings(words, model_path)
        to_cache = []
        for word in words:
            matches = cache[cache['keys'].str.contains(word)]
            print(word, matches.shape)
            if matches.shape[0] < 1:
                to_cache.append(word)
            if matches.shape[0] > 1:
                print(
                    "Warning: Multiple matches in cache for {}. Choose an expression with a unique match.".format(word))
        EmbeddingCache.cache_embeddings(to_cache, model_path)
        return EmbeddingCache.regex_df_column(words, cache, 'keys').values

    @staticmethod
    def _get_pickle_path(path):
        return '.cache/{}.pkl'.format(os.path.splitext(os.path.basename(path))[0])
