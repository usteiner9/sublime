import os
import random

import numpy as np
from .preprocessing import CoNLLTokenizer, SpacyTextTokenizer, JSONLTokenizer
from .preprocessing import Preprocessor
from abc import ABC, abstractmethod
import pandas as pd
import io, json
from .fileutils import FileUtils
from deep_nlp.utils import defaults


class Corpus:

    def __init__(self, path, format='conll'):
        self.data_path = path
        self.format = format
        if not self.format == 'text':
            self.train = os.path.join(defaults.PACKAGEPATH, os.path.join(self.data_path, 'train'))
            self.valid = os.path.join(defaults.PACKAGEPATH, os.path.join(self.data_path, 'valid'))
            self.test = os.path.join(defaults.PACKAGEPATH, os.path.join(self.data_path, 'test'))
        if self.format == 'text':
            self.test = os.path.join(defaults.PACKAGEPATH, os.path.join(self.data_path, 'test'))
        if not os.path.exists(self.test):
            # Look for test files at top level of data
            self.test = self.data_path

    def get_files(self, corpus):
        if corpus == 'train':
            self.train = FileUtils.patch_path(str(self.train))
            return [os.path.join(self.train, x) for x in os.listdir(self.train)]
        elif corpus == 'valid':
            self.valid = FileUtils.patch_path(str(self.valid))
            return [os.path.join(self.valid, x) for x in os.listdir(self.valid)]
        elif corpus == 'test':
            self.test = FileUtils.patch_path(str(self.test))
            return [os.path.join(self.test, x) for x in os.listdir(self.test)]

    def generate_text(self):
        if self.format in ['text', 'conll']:
            for file in self.all_files():
                if not os.path.isdir(file):
                    with io.open(file, encoding='utf-8') as fl:
                        yield fl.read()

        if self.format in ['text', 'conll']:
            for file in self.all_files():
                if not os.path.isdir(file):
                    with io.open(file, encoding='utf-8') as fl:
                        yield fl.read()

        if self.format == 'jsonl':
            for file in self.all_files():
                with open(file) as f:
                    for line in f:
                        yield json.loads(line)[0]

    @property
    def get_format(self):
        return self.format

    def all_files(self):
        return self.get_files('train')+self.get_files('valid')+self.get_files('test')

    def __call__(self, sub_corpus, **kwargs):
        files = self.get_files(sub_corpus)

        #TODO correct this: select num_records file from the corpus
        if 'num_records' in kwargs:
            from random import randrange
            random_index = randrange(0, len(files))
            files = files[random_index]
        return (files, self.format)


class DataFetcher:


    @staticmethod
    def create_matrices(corpus, features):
        all_file_feats = []
        all_file_sents = []

        files = corpus[0]
        format = corpus[1]
        tokenizer = None

        if format == 'text':
            tokenizer = SpacyTextTokenizer()
        if format == 'conll':
            tokenizer = CoNLLTokenizer
        if format == 'jsonl':
            tokenizer = JSONLTokenizer()

        for file in files:
            file_feats = []

            gen = tokenizer.tokenize(file, nlog=1000)
         
            sentences_string = []
            for sentence, sentence_tokenized, label in gen:
                sentences_string.append(sentence_tokenized)
                sentence_feats = []
                for feature, embedding in features.items():
                    if feature == 'contextual_string_embeddings':
                        pass  # Is computed separately
                    elif feature == 'label_embeddings':
                        sentence_feats.append(embedding.embed(label))
                    else:
                        sentence_feats.append(embedding.embed(sentence_tokenized))
                file_feats.append(sentence_feats)
            all_file_feats.append(file_feats)
            all_file_sents.append(sentences_string)
   
        lens = [len(l) for l in all_file_feats]
        all_file_feats = [j for i in all_file_feats for j in i]
        all_file_sents = [j for i in all_file_sents for j in i]
        all_file_feats = Preprocessor.convert_to_dict(all_file_feats, features)
        return all_file_feats, all_file_sents, lens


class BatchGenerator(ABC):

    @abstractmethod
    def generate_batch(self):
        pass


class KerasBatchGenerator(BatchGenerator):

    def __init__(self, data, preprocessing_params, preserve_order=False):

        self.data = data
        self.preprocessing_params = preprocessing_params
        self.preserve_order = preserve_order
        self.shuffle = self.preprocessing_params.get('shuffle', False)
        if self.preserve_order and (self.shuffle or preprocessing_params['batching'] != 'fixed'):
            raise NotImplementedError('Cannot preserve order of sentences with the settings given')
        self.number_of_batches, self.groups = self._group(data, preprocessing_params)

    def _group(self, data, preprocessing_params):
        df = pd.DataFrame(data['word_embeddings'])
        df['sent_len'] = (df > 0).sum(axis=1)
        if preprocessing_params['batching'] == 'sentence_length':
            groups = df.groupby('sent_len')
            return len(groups), groups
        elif preprocessing_params['batching'] == 'fixed':
            batch_size = preprocessing_params['batch_size']
            if not self.preserve_order:  # Sort by length
                df['initial_index'] = df.index  # Keep to restore later
                df.sort_values('sent_len', inplace=True, ascending=False)
                df.reset_index(drop=True, inplace=True)
            # Add column that groups a fixed number of rows into batches
            df['group_index'] = df.index.map(lambda n: n - (n % batch_size))
            if not self.preserve_order:
                df.sort_values('initial_index', inplace=True)
                df.reset_index(drop=True, inplace=True)  # Restore initial index
            groups = df.groupby('group_index')
            return len(groups), groups
        elif preprocessing_params['batching'] == 'fixed_area':
            batch_area = preprocessing_params['batch_area']
            df['initial_index'] = df.index
            df.sort_values('sent_len', inplace=True, ascending=False)
            df.reset_index(drop=True, inplace=True)
            current_area = 0
            current_group_index = 0
            for i, row in df.iterrows():
                sent_len = row['sent_len']
                if current_area + sent_len <= batch_area:
                    current_area += sent_len
                else:
                    current_group_index += 1
                    current_area = sent_len
                df.at[i, 'group_index'] = current_group_index
            df.sort_values('initial_index', inplace=True)
            df.reset_index(drop=True, inplace=True)
            groups = df.groupby('group_index')
            return len(groups), groups
        else:
            raise NotImplementedError()

    def _get_indices_in_group(self, df_group):
        indices = list(df_group.index)
        if self.shuffle:
            random.shuffle(indices)
        return indices

    def generate_batch(self, reset=True):
        self.groups = list(self.groups)
        if self.shuffle:
            random.shuffle(self.groups)

        if self.preprocessing_params['batching'] == 'sentence_length':
            """
            group sentences with same length to create a batch. This avoids padding overhead later.
            """

            while True:
                for sent_len, group in self.groups:
                    if sent_len > 0:
                        indices = self._get_indices_in_group(group)
                        batch = []
                        for k in self.data:
                            if k == 'label_embeddings':
                                batch.append(np.array([self.data[k][idx] for idx in indices])[..., np.newaxis])
                            else:
                                batch.append(np.array([self.data[k][idx] for idx in indices]))

                        # this assumes that labels are the last features in a batch
                        #print(batch[0], batch[1], batch[2], batch[3])
                        yield (batch[0:-1], batch[-1])
                if not reset:
                    break
        elif self.preprocessing_params['batching'] in ['fixed', 'fixed_area']:
            while True:
                for _, group in self.groups:
                    indices = self._get_indices_in_group(group)
                    batch = []
                    for k in self.data:
                        unpadded_rows = [self.data[k][idx] for idx in indices]
                        max_sent_length = len(max(unpadded_rows, key=len))
                        padded_rows = []
                        padded_shape = list(np.array(unpadded_rows[0]).shape)
                        padded_shape[0] = max_sent_length
                        if k == 'word_embeddings':
                            pad_index = self.preprocessing_params['pad_index']['pad_word']
                        elif k == 'char_embeddings':
                            pad_index = self.preprocessing_params['pad_index']['pad_char']
                        elif k == 'case_embeddings':
                            pad_index = self.preprocessing_params['pad_index']['pad_case']
                        elif k == 'label_embeddings':
                            pad_index = self.preprocessing_params['pad_index']['pad_label']
                        else:
                            pad_index = 0
                        for unpadded_row in unpadded_rows:
                            padded_row = np.ones(padded_shape) * pad_index
                            unpadded_array = np.array(unpadded_row)
                            padded_row[0:unpadded_array.shape[0]] = unpadded_array
                            padded_rows.append(padded_row)
                        padded_matrix = np.array(padded_rows)
                        if k == 'label_embeddings':
                            padded_matrix = padded_matrix[..., np.newaxis]
                        batch.append(padded_matrix)

                    # this assumes that labels are the last features in a batch
                    yield (batch[0:-1], batch[-1])
                if not reset:
                    break
        else:
            raise NotImplementedError()

    def num_batches(self):
        print('num_batches: ', self.number_of_batches)
        return self.number_of_batches


def slice_dict(d, num):
    for k, v in d.items():
        d[k] = v[:num]
    return d
