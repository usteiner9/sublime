import numpy as np
import os
from deep_nlp.ner.ner.embeddings import WordEmbedding, CharEmbedding, CaseEmbedding, LabelEmbedding
from deep_nlp.ner.ner.training_utils import Corpus, DataFetcher, KerasBatchGenerator, slice_dict
from deep_nlp.ner.ner.vocab import CaseVocab, CharacterVocab, LabelVocab, WordVocab
from deep_nlp.ner.ner.fileutils import FileUtils
from deep_nlp.ner.ner.preprocessing import Preprocessor
from deep_nlp.ner.ner.trainer import KerasTrainer
from collections import OrderedDict
from deep_nlp.ner.utils import defaults

# set seed fo reproducibility
np.random.seed(42)

config = FileUtils.read_json(os.path.join(defaults.PACKAGEPATH, 'ner/train_params.json'))

# specify global model folder
global_model_dir = os.path.join(defaults.PACKAGEPATH, config['dirs']['models'])

model_name = config['name']['model']
curr_model_dir = os.path.join(global_model_dir, model_name)
FileUtils.create_dir([global_model_dir, curr_model_dir])

# specify corpus folder
corpus = Corpus(config['dirs']['corpus'], format=config['corpus_format'])

# specify directories
char_vocab_path = config['vocab']['char']
word_vocab_path = config['vocab']['word']
case_vocab_path = config['vocab']['case']
label_vocab_path = config['vocab']['label']
dirs = [os.path.join(curr_model_dir,  char_vocab_path), os.path.join(curr_model_dir, case_vocab_path),
        os.path.join(curr_model_dir, label_vocab_path)]
FileUtils.create_dir(dirs)

# generate vocabulary: specify flag
if not config['resume_training']:

    print('generating char vocab')
    CharacterVocab.generate_vocab(corpus, os.path.join(curr_model_dir, char_vocab_path))
    print('generating label vocab')
    LabelVocab.generate_vocab(corpus, os.path.join(curr_model_dir, label_vocab_path))
    print('generating word vocab')
    WordVocab.generate_vocab(os.path.join(defaults.PACKAGEPATH, word_vocab_path),
                             word_vector_size=config['vocab']['word_vector_size'])
    print('generating case vocab')
    CaseVocab.generate_vocab(os.path.join(curr_model_dir, case_vocab_path))

# specify features: These embeddings use the vocab (created or loaded) to embed the documents.
print('creating embeddings')
word_embeddings = WordEmbedding(os.path.join(defaults.PACKAGEPATH, word_vocab_path))  # word vectors are loaded from the embeddings folder
char_embeddings = CharEmbedding(os.path.join(curr_model_dir, char_vocab_path))
case_embeddings = CaseEmbedding(os.path.join(curr_model_dir, case_vocab_path))
label_embeddings = LabelEmbedding(os.path.join(curr_model_dir, label_vocab_path))

# create features for each file in corpus
gen_train_embedded = True
gen_valid_embedded = True
save_feats = True
global_embedded_path = config['dirs']['global_embedded']
curr_embedded_path = os.path.join(global_embedded_path, config['name']['curr_embedded'])
FileUtils.create_dir([global_embedded_path, curr_embedded_path])

features = OrderedDict()
features['word_embeddings'] = word_embeddings
features['char_embeddings'] = char_embeddings
features['case_embeddings'] = case_embeddings
features['label_embeddings'] = label_embeddings


print('generating features')
if gen_train_embedded:
    train_embedded, _, _ = DataFetcher.create_matrices(corpus=corpus('train'), features=features)
    if save_feats:
        FileUtils.save_obj(train_embedded, os.path.join(curr_embedded_path, 'train_embedded.pkl'))
else:
    train_embedded = FileUtils.load_obj(os.path.join(curr_embedded_path, 'train_embedded.pkl'))

if gen_valid_embedded:
    valid_embedded, _, _ = DataFetcher.create_matrices(corpus=corpus('valid'), features=features)
    if save_feats:
        FileUtils.save_obj(valid_embedded, os.path.join(curr_embedded_path, 'valid_embedded.pkl'))
else:
    valid_embedded = FileUtils.load_obj(os.path.join(curr_embedded_path, 'valid_embedded.pkl'))


# pre-processing: cutting, batching, padding
print('preprocessing..')
preprocessing_params = config['preprocessing_params']
print('cutting sentences')
if preprocessing_params['cut_sentence']:
    train_embedded = Preprocessor.cut_sentences(train_embedded, preprocessing_params['max_sentence_length'])
    valid_embedded = Preprocessor.cut_sentences(valid_embedded, preprocessing_params['max_sentence_length'])

print('cutting words')
if preprocessing_params['cut_word']:
    train_embedded = Preprocessor.cut_words(train_embedded,
                                            preprocessing_params['max_word_length'],
                                            preprocessing_params['pad_index']['pad_char'])
    valid_embedded = Preprocessor.cut_words(valid_embedded,
                                            preprocessing_params['max_word_length'],
                                            preprocessing_params['pad_index']['pad_char'])

# train model
print('training')
architecture = 'lstm_cnn'
config['architecture'] = architecture
FileUtils.save_json(config, os.path.join(curr_model_dir, 'config.json'))
keras_trainer = KerasTrainer(path=curr_model_dir,
                             features=features,
                             config=config,
                             batch_generator=KerasBatchGenerator,
                             architecture=architecture)

samples_to_train = None
if samples_to_train:
    train_embedded = slice_dict(train_embedded, samples_to_train)

keras_trainer.train(train_data=train_embedded,
                    valid_data=valid_embedded,
                    epochs=200,
                    resume_training=config['resume_training'],
                    verbose=2)
# score on test corpus: move to a completely different script which can be used by other teams as well.
