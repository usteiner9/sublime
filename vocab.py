from .fileutils import FileUtils
import os
import numpy as np
from abc import ABC, abstractmethod
import io
import spacy
from deep_nlp.utils import defaults


class Vocab(ABC):
    @abstractmethod
    def entity_to_index(self, entity):
        pass

    @abstractmethod
    def index_to_entity(self, index):
        pass

    @abstractmethod
    def exists(self, entity):
        pass

    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def get_matrix(self):
        pass

    @abstractmethod
    def get_path(self):
        pass


class WordVocab(Vocab):

    def shape(self):
        return self.w_v.shape

    def vocab_length(self):
        return len(self.w_i)

    def get_matrix(self):
        return self.w_v

    def __init__(self, path_to_vocab):

        self.path = path_to_vocab
        self.w_i = FileUtils.load_obj(os.path.join(path_to_vocab, 'w_i.pkl'))
        self.i_w = FileUtils.load_obj(os.path.join(path_to_vocab, 'i_w.pkl'))
        self.w_v = None

        if os.path.exists(os.path.join(path_to_vocab, 'w_v.npy')):
            self.w_v = FileUtils.load_npy(os.path.join(path_to_vocab, 'w_v.npy'))

    def entity_to_index(self, word):
        return self.w_i[word] if word in self.w_i else self.w_i[word.lower()]

    def index_to_entity(self, index):
        pass

    def exists(self, word):
        return word.lower() in self.w_i

    def get_path(self):
        return self.path

    @classmethod
    def generate_vocab(cls, path_to_vocab, word_vector_size=300):
        pad_token = '__PAD__'
        unk_token = '__UNK__'

        if os.path.exists(path_to_vocab):
            pass
        else:
            FileUtils.create_dir([path_to_vocab])
            if 'spacy' in os.path.basename(path_to_vocab):
                # SPACY
                vector_size = 300

                w_i = dict()
                i_w = dict()
                w_v = list()

                # PAD vector
                w_i[pad_token] = 0
                i_w[0] = pad_token
                w_v.append(np.zeros((vector_size,)))

                # UNK vector
                w_i[unk_token] = 1
                i_w[1] = unk_token
                w_v.append(np.random.uniform(-0.25, 0.25, vector_size))

                forbidden_tokens = [pad_token, unk_token]
                nlp = spacy.load('en_core_web_lg')
                c_idx = 2
                for lex in nlp.vocab:
                    if lex.has_vector and lex.text not in forbidden_tokens:
                        w_i[lex.text] = c_idx
                        i_w[c_idx] = lex.text
                        w_v.append(lex.vector)
                        c_idx += 1

                del nlp

                FileUtils.save_obj(w_i, os.path.join(path_to_vocab, 'w_i.pkl'))
                FileUtils.save_obj(i_w, os.path.join(path_to_vocab, 'i_w.pkl'))
                FileUtils.save_npy(np.array(w_v), os.path.join(path_to_vocab, 'w_v.npy'))
            else:
                # GLOVE
                # check if a txt file with the name vocab_type exists
                if os.path.exists(path_to_vocab+'.txt'):
                    # if it exists, generate files
                    w_i = dict()
                    i_w = dict()
                    w_v = list()

                    # PAD vector
                    w_i[pad_token] = 0
                    i_w[0] = pad_token
                    w_v.append(np.zeros((word_vector_size,)))

                    # UNK vector
                    w_i[unk_token] = 1
                    i_w[1] = unk_token
                    w_v.append(np.random.uniform(-0.25, 0.25, word_vector_size))

                    forbidden_tokens = [pad_token, unk_token]
                    c_idx = 2

                    with open(path_to_vocab+'.txt', encoding="utf-8") as emb_file:
                        for line in emb_file:
                            split = line.strip().split(" ")
                            word = split[0]
                            if word not in forbidden_tokens:
                                w_i[split[0]] = c_idx
                                i_w[c_idx] = split[0]
                                floats = [float(num) for num in split[1:]]
                                if len(floats) != word_vector_size:
                                    continue
                                w_v.append(np.array([float(num) for num in split[1:]]))
                                c_idx += 1

                    FileUtils.save_obj(w_i, os.path.join(path_to_vocab, 'w_i.pkl'))
                    FileUtils.save_obj(i_w, os.path.join(path_to_vocab, 'i_w.pkl'))
                    FileUtils.save_npy(np.array(w_v), os.path.join(path_to_vocab, 'w_v.npy'))
                else:
                    # else throw error
                    raise Exception


class CharacterVocab(Vocab):

    def shape(self):
        return self.c_v.shape

    def vocab_length(self):
        return len(self.c_i)

    def get_matrix(self):
        return self.c_v

    def __init__(self, path_to_vocab):
        self.path = path_to_vocab
        self.c_i = FileUtils.load_obj(os.path.join(path_to_vocab, 'c_i.pkl'))
        self.i_c = FileUtils.load_obj(os.path.join(path_to_vocab, 'i_c.pkl'))
        self.c_v = None

        if os.path.exists(os.path.join(path_to_vocab, 'c_v.npy')):
            self.c_v = FileUtils.load_npy(os.path.join(path_to_vocab, 'c_v.npy'))

    def entity_to_index(self, char):
        return self.c_i[char]

    def index_to_entity(self, index):
        pass

    def exists(self, char):
        return char in self.c_i

    def get_path(self):
        return self.path

    @staticmethod
    def generate_vocab(corpus, path_to_vocab):
        """
        param files: list of input filenames
        out char_indices: dictionary of characters mapped to indices
        out indices_char: inverse indexing
        """

        data = ''
        text_gen = corpus.generate_text()
        for text in text_gen:
            # if not os.path.isdir(file):
            # with io.open(file, encoding='utf-8') as fl:
            # print(text)
            data += text

        chars = sorted(list(set(data)))
        print('total chars:', len(chars))
        c_i = dict((c, i + 2) for i, c in enumerate(chars))
        i_c = dict((i + 2, c) for i, c in enumerate(chars))

        c_i['PAD'] = 0
        i_c[0] = 'PAD'
        c_i['UNK'] = 1
        i_c[1] = 'UNK'

        FileUtils.save_obj(c_i, os.path.join(path_to_vocab, 'c_i.pkl'))
        FileUtils.save_obj(i_c, os.path.join(path_to_vocab, 'i_c.pkl'))
        # return c_i, i_c


class CaseVocab(Vocab):
    def get_matrix(self):
        return self.c_v

    def shape(self):
        return self.c_v.shape

    def __init__(self, path_to_vocab):
        self.path = path_to_vocab
        self.c_i = FileUtils.load_obj(os.path.join(path_to_vocab, 'c_i.pkl'))
        self.i_c = FileUtils.load_obj(os.path.join(path_to_vocab, 'i_c.pkl'))
        self.c_v = None

        if os.path.exists(os.path.join(path_to_vocab, 'c_v.npy')):
            self.c_v = FileUtils.load_npy(os.path.join(path_to_vocab, 'c_v.npy'))

    def entity_to_index(self, case):
        return self.c_i[case]

    def index_to_entity(self, index):
        pass

    def exists(self, case):
        return case in self.c_i

    def get_path(self):
        return self.path

    @staticmethod
    def generate_vocab(path_to_vocab):
        c_i = {'numeric': 7,
               'allLower': 1,
               'allUpper': 2,
               'initialUpper': 3,
               'other': 4,
               'mainly_numeric': 5,
               'contains_digit': 6,
               'PAD': 0}
        i_c = dict((v, k) for k, v in c_i.items())
        c_v = np.identity(len(c_i), dtype='float32')

        FileUtils.save_obj(c_i, os.path.join(path_to_vocab, 'c_i.pkl'))
        FileUtils.save_obj(i_c, os.path.join(path_to_vocab, 'i_c.pkl'))
        FileUtils.save_npy(c_v, os.path.join(path_to_vocab, 'c_v.npy'))
        # return c_i, i_c, c_v


class LabelVocab(Vocab):
    def get_matrix(self):
        pass

    def ent2id(self):
        return self.bio_cats_idx

    def shape(self):
        return np.array(self.bio_cats).shape

    def __init__(self, path_to_vocab):
        self.path = path_to_vocab
        self.bio_cats = FileUtils.load_obj(os.path.join(path_to_vocab, 'bio_cats.pkl'))
        self.bio_cats_idx = FileUtils.load_obj(os.path.join(path_to_vocab, 'bio_cats_idx.pkl'))
        self.idx_bio_cats = FileUtils.load_obj(os.path.join(path_to_vocab, 'idx_bio_cats.pkl'))
        self.cats = FileUtils.load_obj(os.path.join(path_to_vocab, 'cats.pkl'))
        self.cats_idx = FileUtils.load_obj(os.path.join(path_to_vocab, 'cats_idx.pkl'))
        self.idx_cats = FileUtils.load_obj(os.path.join(path_to_vocab, 'idx_cats.pkl'))

    def entity_to_index(self, label):
        return self.bio_cats_idx[label]

    def index_to_entity(self, index):
        return self.idx_bio_cats[index]

    def exists(self, label):
        return label in self.bio_cats_idx

    def get_path(self):
        return self.path

    @staticmethod
    def generate_vocab(corpus, path_to_vocab):
        pad_token = 'O'
        cats = list()
        cats.append(pad_token)
        #cats.append('O')
        if corpus.get_format == 'conll':
            for file in corpus.all_files():
                if not os.path.isdir(file):
                    with io.open(file, encoding='utf-8') as fl:
                        for line in fl:
                            line = line.strip()
                            toks = line.split()
                            if line and (not toks[0] == u'-DOCSTART-' and u'-DOCSTART-' not in toks[0]):
                                cats.append(toks[-1])
            bio_cats = sorted(set(cats))
            bio_cats[bio_cats.index(pad_token)] = bio_cats[0]
            bio_cats[0] = pad_token
            bio_cats_idx = dict(zip(bio_cats, np.arange(len(bio_cats))))
            idx_bio_cats = {v: k for k, v in bio_cats_idx.items()}
            cats = sorted(set(list(b_c.split('-')[-1] for b_c in bio_cats)))
            cats_idx = dict(zip(cats, np.arange(len(cats))))
            idx_cats = {v: k for k, v in cats_idx.items()}
            print(bio_cats, bio_cats_idx, idx_bio_cats, cats, cats_idx, idx_cats)

            FileUtils.save_obj(bio_cats, os.path.join(path_to_vocab, 'bio_cats.pkl'))
            FileUtils.save_obj(bio_cats_idx, os.path.join(path_to_vocab, 'bio_cats_idx.pkl'))
            FileUtils.save_obj(idx_bio_cats, os.path.join(path_to_vocab, 'idx_bio_cats.pkl'))
            FileUtils.save_obj(cats, os.path.join(path_to_vocab, 'cats.pkl'))
            FileUtils.save_obj(cats_idx, os.path.join(path_to_vocab, 'cats_idx.pkl'))
            FileUtils.save_obj(idx_cats, os.path.join(path_to_vocab, 'idx_cats.pkl'))
            # return bio_cats, cats, ent2id
        if corpus.get_format == 'jsonl':
            def get_label(cats, scheme='BIO'):
                all_tags = []
                if scheme == 'BIO':
                    for tag in scheme:
                        if tag is not 'O':
                            tags = [tag + '-' + str(c) if c != 'O' else c for c in cats]
                            all_tags.append(tags)
                all_tags = [item for sublist in all_tags for item in sublist]
                return all_tags

            for file in corpus.all_files():
                if not os.path.isdir(file):
                    data = FileUtils.load_from_jsonl(file)
                    for rec in data:
                        for ent in rec[1]['entities']:
                            cats.append(ent[2])
            cats = sorted(set(cats))
            bio_cats = get_label(cats, scheme='BIO')
            bio_cats = sorted(set(bio_cats))
            bio_cats_idx = dict(zip(bio_cats, np.arange(len(bio_cats))))
            idx_bio_cats = {v: k for k, v in bio_cats_idx.items()}
            cats = sorted(set(list(b_c.split('-')[-1] for b_c in bio_cats)))
            cats_idx = dict(zip(cats, np.arange(len(cats))))
            idx_cats = {v: k for k, v in cats_idx.items()}
            print(bio_cats, bio_cats_idx, idx_bio_cats, cats, cats_idx, idx_cats)

            FileUtils.save_obj(bio_cats, os.path.join(path_to_vocab, 'bio_cats.pkl'))
            FileUtils.save_obj(bio_cats_idx, os.path.join(path_to_vocab, 'bio_cats_idx.pkl'))
            FileUtils.save_obj(idx_bio_cats, os.path.join(path_to_vocab, 'idx_bio_cats.pkl'))
            FileUtils.save_obj(cats, os.path.join(path_to_vocab, 'cats.pkl'))
            FileUtils.save_obj(cats_idx, os.path.join(path_to_vocab, 'cats_idx.pkl'))
            FileUtils.save_obj(idx_cats, os.path.join(path_to_vocab, 'idx_cats.pkl'))