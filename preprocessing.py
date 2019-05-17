from abc import ABC, abstractmethod
import io
from collections import defaultdict, OrderedDict
import spacy
import os
import re
from .fileutils import FileUtils


class Tok(ABC):
    @staticmethod
    @abstractmethod
    def tokenize(file, labels):
        pass


class SentenceTokenizer(Tok):
    @staticmethod
    @abstractmethod
    def tokenize(file, **kwargs):
        pass

"""
class CoNLLTokenizer(SentenceTokenizer):
    @staticmethod
    def tokenize(file, **kwargs):
        sent_count = 0
        if 'nlog' in kwargs:
            nlog = kwargs['nlog']
        else:
            nlog = 1000
        if not os.path.isdir(file):
            with io.open(file, encoding='utf-8') as fl:
                for line in fl:
                    line = line.strip()
                    toks = line.split()
                    # sentence break
                    if len(toks) < 1:
                        if len(sent) > 0:
                            sent_count += 1
                            if sent_count % nlog == 0:
                                print(sent_count, sent)
                            doc = sent
                            words = doc.split(' ')[:-1]
                            word_labels = cats
                            sent = ""
                            cats = []
                            yield doc, words, word_labels

                    # sentence start
                    elif toks[0] == u'-DOCSTART-' or u'-DOCSTART-' in toks[0]:
                        sent = ""
                        cats = []
                    else:
                        sent += toks[0] + ' '
                        cats.append(toks[-1])

"""                       

class CoNLLTokenizer(SentenceTokenizer):
    @staticmethod
    def tokenize(file, **kwargs):
        sent_count = 0
        if 'nlog' in kwargs:
            nlog = kwargs['nlog']
        else:
            nlog = 1000
        if not os.path.isdir(file):
            with io.open(file, encoding='utf-8') as fl:
                for line in fl:
                    line = line.rstrip()
                    toks = line.split()
                    
                    # sentence break
                    if len(toks) < 1:
                        if len(sent) > 0:
                            sent_count += 1
                            if sent_count % nlog == 0:
                                print(sent_count, sent)
                            doc = sent
                            #words = doc.split(' ')[:-1]
                            word_labels = cats
                            full_tokens = tokens
                            sent = ""
                            cats = []
                            tokens = []
                            yield doc, full_tokens, word_labels

                    # sentence start
                    elif toks[0] == u'-DOCSTART-' or u'-DOCSTART-' in toks[0]:
                        sent = ""
                        cats = []
                        tokens = []
                    else:
                        word = re.sub(" "+ toks[-1], "", line)
                        sent += word + ' '
                        tokens.append(word)
                        cats.append(toks[-1])

class SpacyTextTokenizer(SentenceTokenizer):

    def __init__(self):
        # self.nlp = spacy.load('en_core_web_lg')
        self.nlp = spacy.load('en')

    def tokenize(self, file, **kwargs):
        # TODO currently it doesn't allow parallel processing. That can speed up the tokenization
        if not os.path.isdir(file):
            if file.endswith('.txt'):
                with open(file) as f:
                    content = f.readlines()

                    content = [x.strip() for x in content]
                    content = list(filter(None, content))
                    content = ' '.join(content)
                    doc = self.nlp(content)
                    for sent in doc.sents:
                        yield sent.text, [tok.text for tok in list(sent)], []


class JSONLTokenizer(SentenceTokenizer):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def tokenize(self, file, **kwargs):
        def between(query, all_points):
            for point, val in all_points.items():
                if query[0] > point[0] and query[1] < point[1]:
                    return True, val
            return False, None
        data = FileUtils.load_from_jsonl(file)
        for rec_idx, rec in enumerate(data):
            doc_nlp = self.nlp(rec[0])
            ents = dict()
            for ent in rec[1]['entities']:
                ents[(ent[0], ent[1])] = ent[2]

            doc_tok = [tok for tok in doc_nlp]
            labels = []
            for tok in doc_tok:
                b, c = between((tok.idx, tok.idx + len(tok.text)), ents)
                if tok.idx in [offset[0] for offset in list(ents.keys())] and tok.idx + len(
                        tok.text) in [offset[1] for offset in list(ents.keys())]:
                    labels.append('B-'+ents[(tok.idx, tok.idx+len(tok.text))])
                elif tok.idx in [offset[0] for offset in list(ents.keys())]:
                    for k, v in ents.items():
                        if k[0] == tok.idx:
                            labels.append('B-'+v)
                elif tok.idx + len(tok.text) in [offset[1] for offset in list(ents.keys())]:
                    for k, v in ents.items():
                        if k[1] == tok.idx + len(tok.text):
                            labels.append('I-'+v)
                elif b:
                    labels.append('I-'+c)
                else:
                    labels.append('O')
            yield doc_nlp.text, [t.text for t in doc_tok], labels


class Preprocessor:

    def window(iterable, size):
        res = []
        for i_idx in range(1, len(iterable) -1):
        #for i_idx in range(0, len(iterable) ): #Uncomment this to add blank sentence before and after the first and last sentence of the corpus in the sliding window
            window = []
            for j in range(i_idx-size, i_idx+size +1 ):
                #Add spaces to edge cases
                if j<0 or j>len(iterable) -1:
                    #window +=[' ']
                    pass
                else:
                    window+=iterable[j]
            res.append(window)
        return res

    @classmethod
    def overlap(cls,iterable):
        size = 1 # To be discussed whether the window size will always be the previous and following sentence or several sentences. (Waiting for exp. results.)
        if type(iterable) == list:
            iterable = cls.window(iterable, size)
        else:
            for f in iterable:
                iterable[f] = cls.window(iterable[f], size)
        return iterable
    
    @classmethod
    def cut_sentences(cls, feats, max_sentence_length):

        for f in feats:
            feats[f] = [w[:max_sentence_length] for w in feats.get(f)]

        return feats

    @classmethod
    def cut_words(cls, feats, max_word_length, pad_char):
        if 'char_embeddings' in feats:
            feats['char_embeddings'] = [[w[:max_word_length] if len(w) > max_word_length else w+[pad_char]*(max_word_length-len(w)) for w in k] for k in feats.get('char_embeddings')]
        return feats

    @classmethod
    def convert_to_dict(cls, feats, features):
        feats_dict = OrderedDict()
        for sent_vector in feats:
            for i, f in enumerate(features):
                if f not in feats_dict:
                    feats_dict[f] = []
                feats_dict[f].append(sent_vector[i])
        return feats_dict

    @classmethod
    def pad(cls, data, max_sequence_length, pad_index, batching):
        if batching == 'fixed':
            for k, v in data.items():
                if k == 'word_embeddings':
                    pad_token = pad_index.get('pad_word')
                    data[k] = [sent + [pad_token] * (max_sequence_length - len(sent)) for sent in v]
                if k == 'char_embeddings':
                    pad_token = pad_index.get('pad_char')
                    max_word_length = len(v[0][0])
                    data[k] = [sent + [[pad_token] * max_word_length] * (max_sequence_length - len(sent)) for sent in v]
                if k == 'case_embeddings':
                    pad_token = pad_index.get('pad_case')
                    data[k] = [sent + [pad_token] * (max_sequence_length - len(sent)) for sent in v]

                if k == 'label_embeddings':
                    pad_token = pad_index.get('pad_label')
                    data[k] = [sent + [pad_token] * (max_sequence_length - len(sent)) for sent in v]
        return data
