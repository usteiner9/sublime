from .embeddings import WordEmbedding, CharEmbedding, CaseEmbedding, LabelEmbedding
from .preprocessing import Preprocessor
import os
from ..utils.spacy import tokenentlists2spacydoc
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import spacy
from ..utils import defaults


def _get_embeddings(type_, vocab, model_dir):
    embedding = None
    if type_ == 'word_embeddings':
        embedding = WordEmbedding(os.path.join(defaults.PACKAGEPATH, vocab['word']))
    if type_ == 'char_embeddings':
        embedding = CharEmbedding(os.path.join(model_dir, vocab['char']))
    if type_ == 'case_embeddings':
        embedding = CaseEmbedding(os.path.join(model_dir, vocab['case']))
    if type_ == 'label_embeddings':
        embedding = LabelEmbedding(os.path.join(model_dir, vocab['label']))
    return embedding

def parse_features(config, model_dir):
    features = OrderedDict()
    for k in config['features']:
        features[k] = _get_embeddings(k, config['vocab'], model_dir)
    # labels = features.pop('label_embeddings', None)
    return features


def preprocess(data, preprocessing_params):
    if 'cut_sentence' in preprocessing_params:
        data = Preprocessor.cut_sentences(data, preprocessing_params['max_sentence_length'])
    if 'cut_word' in preprocessing_params:
        data = Preprocessor.cut_words(data,
                                      preprocessing_params['max_word_length'],
                                      preprocessing_params['pad_index']['pad_char'])
    return data


def truncate(sentences, preds, missing_label_index):
    # Note, preds at the sentence level always have length set in the model config file (see max_sentence_length)
    # This function matches the pred length (at the sentence) level to the actual size of the sentence:
    for idx in range(len(preds)):

        len_sentence = len(sentences[idx])
        len_pred = len(preds[idx])

        # Truncate Padding - case when sentence is smaller than prediction:
        if len_sentence <= len_pred:
            preds[idx] = preds[idx][:len_sentence]
        # Add Padding - case when sentence is longer than the prediction:
        else:
            preds[idx] = np.append(preds[idx], np.ones(len_sentence-len_pred)*missing_label_index)

        assert len(preds[idx]) == len_sentence, "Error in prediction resizing method, test_utils.truncate"
    return preds


def decode(preds, label_embeddings):

    preds = [label_embeddings.index_to_entity(i) for i in preds]
    return preds


def label(preds, label_embeddings):
    labelled_sents = []
    for pred in preds:
        labelled = decode(pred, label_embeddings)
        labelled_sents.append(labelled)
    return labelled_sents


def format_output_raw(preds, test_sentences_tokenized, sent_lens):
    """
    Returns lists formatting for the scored results:

    Parameters
    ----------
    preds
    test_sentences_tokenized
    sent_lens

    Returns
    -------
    doc_sent_token_preds    : list[list[list]]
        list of lists of lists containing predicted labels for each token, in each sentence, in each document,
        respectively.
    doc_sent_tokens         : list[list[list]]
        list of lists of lists containing tokenized parts for each sentence, for each document,
        respectively.

    """
    i = 0
    doc_sent_preds = []  # document wise predictions
    doc_sent_tokens = []  # document wise predictions
    for n in sent_lens:
        doc_preds = []
        doc_sents = []
        doc_preds.append(preds[i:i+n])
        doc_sents.append(test_sentences_tokenized[i:i+n])
        i += n
        doc_preds = [j for i in doc_preds for j in i]
        doc_sents = [j for i in doc_sents for j in i]
        doc_sent_preds.append(doc_preds)
        doc_sent_tokens.append(doc_sents)

    return doc_sent_preds, doc_sent_tokens


def format_output_spacy(preds, test_sentences_tokenized, sent_lens, entity_vocab):
    """
    Returns spacy document objects for the scored results:

    Parameters
    ----------
    preds
    test_sentences_tokenized
    sent_lens
    entity_vocab               : spacy.

    Returns
    -------
    spacy_docs  : list[spacy.tokens.doc.Doc]
        A list of spacy objects containing text and named entities
    """

    # Get lists of outputs:
    doc_sent_preds, doc_sent_tokens = format_output_raw(preds, test_sentences_tokenized, sent_lens)

    # Load light-weight spacy model for output:
    # nlp = spacy.load("en_core_web_lg")
    # nlp = spacy.load("en")
    nlp = spacy.blank("en")

    # Brute Force, add all entities (including BILUO) to vocab:
    for ent in entity_vocab.ent2id().keys():
        # Add full entity:
        nlp.vocab.strings.add(ent)

        # Add entity without BIO term:
        if len(ent[2:]) and ent[1] == "-":
            nlp.vocab.strings.add(ent[2:])

    # Construct a spacy objects for each doc:
    spacy_docs = []
    for doc_idx, doc in tqdm(
            enumerate(doc_sent_tokens), desc="constructing spacy documents...", total=len(doc_sent_tokens)
    ):
        # Construct spacy doc with constructor util:
        spacy_docs += [tokenentlists2spacydoc(doc, doc_sent_preds[doc_idx], nlp)]

    return spacy_docs
