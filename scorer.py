from abc import ABC, abstractmethod
import keras
import numpy as np
import os
from .layers import CRF


class Loader(ABC):
    @abstractmethod
    def load_model(self, model_path):
        pass


class KerasLoader(Loader):
    @classmethod
    def load_model(cls, model_dir, **kwargs):
        if 'use_crf' in kwargs:
            def create_custom_objects():
                instanceHolder = {"instance": None}

                class ClassWrapper(CRF):
                    def __init__(self, *args, **kwargs):
                        instanceHolder["instance"] = self
                        super(ClassWrapper, self).__init__(*args, **kwargs)

                def loss(*args):
                    method = getattr(instanceHolder["instance"], "loss_function")
                    return method(*args)

                def accuracy(*args):
                    method = getattr(instanceHolder["instance"], "accuracy")
                    return method(*args)

                return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "loss": loss, "accuracy": accuracy}
            model = keras.models.load_model(os.path.join(model_dir, 'model.h5'), custom_objects=create_custom_objects())
        else:
            model = keras.models.load_model(os.path.join(model_dir, 'model.h5'))

        return model


class Predictor(ABC):
    @classmethod
    @abstractmethod
    def predict(cls, model, data, preprocessing_params):
        pass


class KerasPredictor(Predictor):
    @classmethod
    def predict(cls, model, test_embedded, preprocessing_params, batch_size=512):
        from deep_nlp.ner.ner.training_utils import KerasBatchGenerator
        preprocessing_params['batching'] = 'fixed'
        preprocessing_params['shuffle'] = False
        preprocessing_params['batch_size'] = batch_size
        test_gen = KerasBatchGenerator(test_embedded, preprocessing_params, preserve_order=True)
        preds = []
        for features, _ in test_gen.generate_batch(reset=False):
            pred = model.predict_on_batch(features)
            preds += list(pred.argmax(axis=-1))
        return np.array(preds)
