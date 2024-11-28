import os
import opusfilter
from opusfilter.util import grouper
import itertools
import collections
from scipy.spatial.distance import cosine
import string
import re
from collections import Counter

import pickle
from laser_encoders import LaserEncoderPipeline
from align_gpt import AlignGPT

class RingFilter(opusfilter.FilterABC):
    score_direction = opusfilter.CLEAN_LOW
    accept_threshold = 1 + 10**-6
    reject_threshold = 0

    def __init__(self, threshold=0.33, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def ring_ratio(self, sentence):
        words = sentence.split()
        length = len(words)
        if length > 0:
            word_count = Counter(words)
            return word_count["ring"] / length
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self.ring_ratio(sentence) for sentence in pair]

    def accept(self, score):
        return all(ratio <= self.threshold for ratio in score)


class IdenticalStringsFilter(opusfilter.FilterABC):
    score_direction = opusfilter.CLEAN_FALSE
    accept_threshold = 1
    reject_threshold = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            yield [pair[0] == pair[1]]

    def accept(self, score):
        return not any(score)


class LaserEmbeddingFilter(opusfilter.FilterABC):
    """Filtering based on multilingual sentence embeddings"""

    score_direction = opusfilter.CLEAN_HIGH
    accept_threshold = 0
    reject_threshold = 1 + 10**-6

    def __init__(self, languages=None, threshold=0.5, nn_model=None, chunksize=200, **kwargs):
        # if languages is None:
            # raise ConfigurationError("A list of language codes needs to be defined")
        self.threshold = threshold
        self.languages = languages
        self.embedding_model = {lang: LaserEncoderPipeline(lang=lang) for lang in languages}
        self.nn_model = None
        self.chunksize = chunksize
        super().__init__(**kwargs)
        if nn_model:
            with open(os.path.join(self.workdir, nn_model), 'rb') as fobj:
                self.nn_model = pickle.load(fobj)

    def _cosine_similarities(self, pairs):
        """Calculate cosine similarities for the segments"""
        chunksize = len(pairs)
        segments = [segment for pair in pairs for segment in pair]
        languages = self.languages * chunksize
        embeddings = [self.embedding_model[lang].encode_sentences([seg]) for lang, seg in zip(languages, segments)]
        for pos in range(0, len(languages), len(self.languages)):
            yield [1 - cosine(embeddings[pos + idx1][0], embeddings[pos + idx2][0])
                   for idx1, idx2 in itertools.combinations(range(len(self.languages)), 2)]

    @staticmethod
    def _ratio_normalize(vec1, vec2, n_neighbors, nn_sum1, nn_sum2):
        """Cosine similarity normalized by similarity to nearest neighbors"""
        return 2 * n_neighbors * (1 - cosine(vec1, vec2)) / (nn_sum1 + nn_sum2)

    def _normalized_similarities(self, pairs):
        """Calculate normalized cosine similarities for the segments"""
        chunksize = len(pairs)
        n_neighbors = self.nn_model.n_neighbors
        input_per_lang = zip(*pairs)
        output_per_lang = []
        nn_sums = collections.defaultdict(dict)
        for idx, segments in enumerate(input_per_lang):
            for other_idx, other_language in enumerate(self.languages):
                if idx == other_idx:
                    continue
                dists, _ = self.nn_model.query(segments, other_language)
                nn_sums[idx][other_idx] = dists.sum(axis=1)
            embeddings = self.embedding_model[self.languages[idx]].encode_sentences(segments)
            output_per_lang.append(embeddings)
        for pos in range(chunksize):
            yield [self._ratio_normalize(output_per_lang[idx1][pos], output_per_lang[idx2][pos],
                                         n_neighbors, nn_sums[idx1][idx2][pos], nn_sums[idx2][idx1][pos])
                   for idx1, idx2 in itertools.combinations(range(len(self.languages)), 2)]

    def _score_chunk(self, chunk):
        """Return scores for a chunk of data"""
        return self._cosine_similarities(chunk) if self.nn_model is None else \
            self._normalized_similarities(chunk)

    def score(self, pairs):
        for chunk in grouper(pairs, self.chunksize):
            for score in self._score_chunk(chunk):
                yield score

    def accept(self, score):
        return all(similarity >= self.threshold for similarity in score)

class PunctuationFilter(opusfilter.FilterABC):
    """Filter based on the ratio of punctuation characters in sentences"""

    score_direction = opusfilter.CLEAN_LOW
    accept_threshold = 1 + 10**-6
    reject_threshold = 0

    def __init__(self, threshold=0.2, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def punctuation_ratio(self, sentence):
        length = len(sentence)
        if length > 0:
            num_punctuation = sum(1 for char in sentence if char in string.punctuation)
            return num_punctuation / length
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self.punctuation_ratio(sentence) for sentence in pair]

    def accept(self, score):
        return all(ratio < self.threshold for ratio in score)

class NumberFilter(opusfilter.FilterABC):
    """Filter based on the ratio of numerical characters in sentences"""

    score_direction = opusfilter.CLEAN_LOW
    accept_threshold = 1 + 10**-6
    reject_threshold = 0

    def __init__(self, threshold=0.2, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def number_ratio(self, sentence):
        length = len(sentence)
        if length > 0:
            num_numbers = sum(1 for char in sentence if char.isdigit())
            return num_numbers / length
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self.number_ratio(sentence) for sentence in pair]

    def accept(self, score):
        return all(ratio < self.threshold for ratio in score)


class AlignGPTCleaner(opusfilter.PreprocessorABC):
    """Tokenize text"""

    def __init__(self, source_language, target_language, batch_size=16, **kwargs):
        self.align_gpt = AlignGPT(batch_size=batch_size)
        self.source_language = source_language
        self.target_language = target_language
        super().__init__(**kwargs)

    def process(self, pairs):
        pairs = list(pairs)
        cleaned_pairs, valid_flags = self.align_gpt.run_batch_alignment(pairs, source_language=self.source_language, target_language=self.target_language)
        for i in range(len(valid_flags)):
            if valid_flags[i]:
                yield cleaned_pairs[i]
