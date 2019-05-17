import os
from .training_utils import Corpus, DataFetcher
from .fileutils import FileUtils
from .test_utils import parse_features, preprocess, truncate, label
from .scorer import KerasLoader, KerasPredictor
from .callbacks import compute_f1

# specify corpus folder
corpus = Corpus('../resources/data/score/score_energia', format='text')

# specify model folder
global_model_dir = '../models'
model_name = 'lstmcnncrf_2008_checkfiles'
model_dir = os.path.join(global_model_dir, model_name)
config = FileUtils.read_json(os.path.join(model_dir, 'config.json'))

# parse config file
features = parse_features(config['features'])

# create features for each file in corpus
print('generating features')
test_embedded, test_sentences_tokenized, sent_lens = DataFetcher.create_matrices(corpus=corpus('test'),
                                                                                 features=features)
FileUtils.save_obj(test_embedded, os.path.join(model_dir, 'test_embedded.pkl'))
FileUtils.save_obj(test_sentences_tokenized, os.path.join(model_dir, 'test_sentences_tokenized.pkl'))
FileUtils.save_obj(sent_lens, os.path.join(model_dir, 'sent_lens.pkl'))

# pre-process test data
test_embedded = preprocess(test_embedded, config['preprocessing_params'])

# loading the model
print('loading model')
model = KerasLoader.load_model(model_dir, use_crf=config['use_crf'])

print('making predictions')
labels = test_embedded.pop('label_embeddings', None)
preds = KerasPredictor.predict(model, test_embedded)
preds = truncate(test_sentences_tokenized, list(preds))
preds = label(preds, features['label_embeddings'])

print('preds: ', preds)
print('sentences: ', test_sentences_tokenized)
FileUtils.save_obj(preds, os.path.join(model_dir, 'test_sentences_preds.pkl'))

if labels:
    # compute f1
    labels = truncate(test_sentences_tokenized, list(labels))
    labels = label(labels, features['label_embeddings'])
    p, r, f = compute_f1(preds, labels)
    print(p, r, f)

i = 0
documents_preds = [] # document wise predictions
documents_sents = [] # document wise predictions
for n in sent_lens:
    doc_preds = []
    doc_sents = []
    doc_preds.append(preds[i:i+n])
    doc_sents.append(test_sentences_tokenized[i:i+n])
    i+=n
    doc_preds = [j for i in doc_preds for j in i]
    doc_sents = [j for i in doc_sents for j in i]
    documents_preds.append(doc_preds)
    documents_sents.append(doc_sents)






