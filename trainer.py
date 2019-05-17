from abc import ABC, abstractmethod
from keras.callbacks import CSVLogger
import os
from .fileutils import FileUtils
from .callbacks import F1_eval, PlotLosses
from .scorer import KerasLoader
from .models import Models


class Trainer(ABC):

    @abstractmethod
    def train(self, train_data,
              valid_data,
              test_data=None,
              epochs=200,
              resume_training=True,
              verbose=0):
        pass


class KerasTrainer(Trainer):
    def __init__(self,
                 path,
                 features,
                 config,
                 batch_generator,
                 architecture='lstm_cnn',
                 cs_embeddings=None):
        self.generator = batch_generator
        self.features = features
        self.config = config
        self.use_crf = config['use_crf']
        assert architecture in Models.supported_models
        self.architecture = architecture
        self.cs_embeddings = cs_embeddings
        self.path = path
        self.train_gen = None
        self.valid_gen = None
        self.test_gen = None
        self.model = None
        self.callbacks = None

    """
    ####### code needs to be re-written with Elmo and FLAIR pull-requests #######
    def _build_config(self):
        config = Config(os.path.join(self.path, 'config.json'))
        features_dict = OrderedDict()
        for k, v in self.features.items():
            try:
                features_dict[k] = v.get_vocab_path()
            except AttributeError:
                pass
        config.set('features', features_dict)
        config.set('architecture', self.architecture)
        config.set('preprocessing_params', self.preprocessing_params)
        config.set('use_crf', self.use_crf)
        if self.cs_embeddings:
            lm_path = os.path.join(self.path, 'lm')
            FileUtils.create_dir([lm_path])
            config.set('lm_forward', os.path.join(lm_path, 'forward.pt'))
            config.set('lm_backward', os.path.join(lm_path, 'backward.pt'))
            copyfile(self.cs_embeddings.forward_path, config.get('lm_forward'))
            copyfile(self.cs_embeddings.backward_path, config.get('lm_backward'))
        config.save()
        return config
    """

    def train(self, train_data, valid_data, test_data=None, epochs=200, resume_training=False, verbose=0):

        self.train_gen = self.generator(train_data,
                                        self.config['preprocessing_params'])
        self.valid_gen = self.generator(valid_data,
                                        self.config['preprocessing_params'])
        if test_data is not None:
            self.test_gen = self.generator(test_data,
                                           self.config['preprocessing_params'])
        if not resume_training:
            print('building model')
            self.model = Models.get_model(self.architecture, self.features, self.config)
        else:
            print('loading model')
            self.model = KerasLoader.load_model(self.path, use_crf=self.use_crf)

        print(self.model.summary())
        FileUtils.save_json(self.model.to_json(), os.path.join(self.path, 'model.json'))

        self.callbacks = self._get_callbacks()

        self.model.fit_generator(self.train_gen.generate_batch(reset=True),
                                 steps_per_epoch=self.train_gen.num_batches(),
                                 epochs=epochs,
                                 validation_data=self.valid_gen.generate_batch(reset=True),
                                 validation_steps=self.valid_gen.num_batches(),
                                 verbose=verbose,
                                 initial_epoch=0,
                                 callbacks=self.callbacks,
                                 max_queue_size=0,
                                 workers=0)

    def _get_callbacks(self):
        evaluated_data = [
            self.valid_gen,
            self.train_gen,
        ]
        if self.test_gen is not None:
            evaluated_data.append(self.test_gen)
        f1_eval = F1_eval(filepath=self.path,
                          validation_data=evaluated_data,
                          label_embeddings=self.features['label_embeddings'],
                          save_model=True)
        csv_logger = CSVLogger(os.path.join(self.path, 'training.csv'))

        plots_dir = os.path.join(self.path, 'plots_dir')
        FileUtils.create_dir([plots_dir])
        plot_loss = PlotLosses(plots_dir)
        callback_list = [f1_eval, csv_logger, plot_loss]
        return callback_list
