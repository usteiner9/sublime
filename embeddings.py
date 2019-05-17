from abc import ABC, abstractmethod
from .vocab import WordVocab, LabelVocab, CharacterVocab, CaseVocab
"""
import wget, os
from .fileutils import FileUtils
from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
from flair.embeddings import CharLMEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence"""


class Embedding(ABC):

    @abstractmethod
    def embed(self, file):
        pass

    @property
    @abstractmethod
    def embedding_type(self):
        pass

    @abstractmethod
    def get_vocab_path(self):
        pass

    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def weights(self):
        pass


"""
# Classes not used at the moment...
class ContextualEmbedding(Embedding):

    def __init__(self, elmo_dir, layers=3, force_download=False):

        self.elmo = None
        self.layers = layers
        if not os.listdir(elmo_dir) or force_download:
            print('downloading elmo model...')
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

            self.elmo = Elmo(options_file, weight_file, self.layers, dropout=0)
            FileUtils.save_obj(self.elmo, os.path.join(elmo_dir, 'elmo.pkl'))
        if not self.elmo:
            self.elmo = FileUtils.load_obj(os.path.join(elmo_dir, 'elmo.pkl'))

    def embed(self, sentence_tokenized):
        sentences = [sentence_tokenized]
        character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)

        layer_reps = []
        for layer in embeddings['elmo_representations'][:self.layers]:
            layer_reps.append(layer.data.numpy())
        output = np.squeeze(np.array(layer_reps).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3), axis=0)
        return list(output)

    @property
    def embedding_type(self):
        return 'elmo_embeddings'

    def get_vocab_path(self):
        pass

    def shape(self):
        return [1024, self.layers]

    def weights(self):
        pass


class FlairEmbedding(Embedding):

    def __init__(self):
        self.charlm_embedding_forward = CharLMEmbeddings('news-forward')
        self.charlm_embedding_backward = CharLMEmbeddings('news-backward')
        self.stacked_embeddings = StackedEmbeddings(embeddings=[self.charlm_embedding_forward,
                                                                self.charlm_embedding_backward])

    def embed(self, sentence_tokenized):
        sentence = ' '.join(sentence_tokenized)
        sentence = Sentence(sentence)
        self.stacked_embeddings.embed(sentence)
        return [token.embedding.data.numpy() for token in sentence]

    @property
    def embedding_type(self):
        return 'flair_embeddings'

    def get_vocab_path(self):
        pass

    def shape(self):
        return [4096]

    def weights(self):
        pass
"""


class WordEmbedding(Embedding):
    def shape(self):
        return self.vocab.shape()

    def get_vocab_path(self):
        return self.vocab.get_path()

    def vocab_length(self):
        return self.vocab.vocab_length()

    @property
    def embedding_type(self):
        return 'word_embeddings'

    def __init__(self, path_to_vocab):
        self.vocab = WordVocab(path_to_vocab)

    def weights(self):
        return self.vocab.get_matrix()

    def embed(self, sentence_tokenized):
        sentence_indexed = []
        for word in sentence_tokenized:
            if self.vocab.exists(word):
                sentence_indexed.append(self.entity_to_index(word))
            else:
                # for backward compatibility
                if '__UNK__' in self.vocab.w_i:
                    sentence_indexed.append(self.entity_to_index('__UNK__'))
                else:
                    sentence_indexed.append(self.entity_to_index('UNK'))
        return sentence_indexed

    def entity_to_index(self, word):
        return self.vocab.entity_to_index(word)


class CharEmbedding(Embedding):
    def shape(self):
        return self.vocab.shape()

    def vocab_length(self):
        return self.vocab.vocab_length()

    def get_vocab_path(self):
        return self.vocab.get_path()

    @property
    def embedding_type(self):
        return 'char_embeddings'

    def __init__(self, path_to_vocab):
        self.vocab = CharacterVocab(path_to_vocab)

    def embed(self, sentence_tokenized):
        sentence_indexed = []
        for word in sentence_tokenized:
            word_indexed = []
            for char in word:
                if self.vocab.exists(char):
                    word_indexed.append(self.entity_to_index(char))
                else:
                    word_indexed.append(self.entity_to_index('UNK'))
            sentence_indexed.append(word_indexed)
        return sentence_indexed

    def entity_to_index(self, word):
        return self.vocab.entity_to_index(word)

    def weights(self):
        return self.vocab.get_matrix()


class CaseEmbedding(Embedding):
    def weights(self):
        return self.vocab.get_matrix()

    def shape(self):
        return self.vocab.shape()

    def get_vocab_path(self):
        return self.vocab.get_path()

    @property
    def embedding_type(self):
        return 'case_embeddings'

    def __init__(self, path_to_vocab):
        self.vocab = CaseVocab(path_to_vocab)

    def embed(self, sentence_tokenized):
        sentence_indexed = []
        for word in sentence_tokenized:
            casing = 'other'
            
            if len(word) == 0: 
                casing = 'other'
            else: 
                numDigits = 0
                for char in word:
                    if char.isdigit():
                        numDigits += 1

                digitFraction = numDigits / float(len(word))

                if word.isdigit():  # Is a digit
                    casing = 'numeric'
                elif digitFraction > 0.5:
                    casing = 'mainly_numeric'
                elif word.islower():  # All lower case
                    casing = 'allLower'
                elif word.isupper():  # All upper case
                    casing = 'allUpper'
                elif word[0].isupper():  # is a title, initial char upper, then all lower
                    casing = 'initialUpper'
                elif numDigits > 0:
                    casing = 'contains_digit'

            if self.vocab.exists(casing):
                sentence_indexed.append(self.entity_to_index(casing))

        return sentence_indexed

    def entity_to_index(self, word):
        return self.vocab.entity_to_index(word)


class LabelEmbedding(Embedding):

    def weights(self):
        pass

    def shape(self):
        return self.vocab.shape()

    def get_vocab_path(self):
        return self.vocab.get_path()

    @property
    def embedding_type(self):
        return 'label_embeddings'

    def __init__(self, path_to_vocab):
        self.vocab = LabelVocab(path_to_vocab)

    def embed(self, sentence_tokenized):
        sentence_indexed = []
        for label in sentence_tokenized:
            if self.vocab.exists(label):
                sentence_indexed.append(self.entity_to_index(label))
            else:
                sentence_indexed.append(self.entity_to_index('O'))
        return sentence_indexed

    def entity_to_index(self, word):
        return self.vocab.entity_to_index(word)

    def index_to_entity(self, index):
        return self.vocab.index_to_entity(index)

