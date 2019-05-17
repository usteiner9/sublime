from keras.layers import Input, Bidirectional, Embedding, TimeDistributed, Dropout, Conv1D, GlobalMaxPooling1D, LSTM, \
    Flatten, concatenate, Dense
from keras.initializers import RandomUniform
from keras import Model
from .layers import CRF


class Models:

    supported_models = ['lstm_lstm', 'lstm_cnn', 'flair', 'elmo', 'hybrid']

    @classmethod
    def get_model(cls, model_name, features, config):
        if model_name in cls.supported_models:
            return cls._build_model(model_name, features, config)

    @classmethod
    def _build_model(cls, model_name, features, config):
        if model_name == 'lstm_lstm':
            word_input = Input(shape=(None,), name='word_input')
            words = Embedding(input_dim=features['word_embeddings'].shape()[0],
                              output_dim=features['word_embeddings'].shape()[1],
                              weights=[features['word_embeddings'].weights()],
                              trainable=False,
                              name='words_embedding')(word_input)

            # build character based embedding
            char_input = Input(shape=(None, config['preprocessing_params']['max_word_length'],), dtype='int32',
                               name='char_input')
            char_embeddings = TimeDistributed(Embedding(input_dim=features['char_embeddings'].vocab_length(),
                                                        output_dim=25,
                                                        name='char_embeddings'
                                                        ))(char_input)

            chars = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False)))(
                char_embeddings)

            # combine characters and word embeddings
            x = concatenate([words, chars])
            x = Dropout(0.5)(x)

            x = Bidirectional(LSTM(units=100,
                                   return_sequences=True,
                                   recurrent_dropout=0.3))(x)
            x = Dropout(0.5)(x)
            x = Dense(100, activation='tanh')(x)
                
            output = Dense(features['label_embeddings'].shape()[0])(x)

            if config['use_crf']:
                crf = CRF(features['label_embeddings'].shape()[0], sparse_target=True)
                output = crf(output)
                model = Model(inputs=[word_input, char_input], outputs=[output])
                model.compile(optimizer='adam', loss=crf.loss_function)
            else:
                model = Model(inputs=[word_input, char_input], outputs=[output])
                model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")
            return model
# Some modification to original version
#         if model_name == 'lstm_cnn':
#             words_input = Input(shape=(None,), dtype='int32', name='words_input')
#             words = Embedding(input_dim=features['word_embeddings'].shape()[0],
#                               output_dim=features['word_embeddings'].shape()[1],
#                               weights=[features['word_embeddings'].weights()],
#                               trainable=False,
#                               name='words_embedding')(words_input)

#             casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
#             case = Embedding(input_dim=features['case_embeddings'].shape()[0],
#                              output_dim=features['case_embeddings'].shape()[0],
#                              weights=[features['case_embeddings'].weights()],
#                              trainable=False,
#                              name='case_embedding')(casing_input)

#             character_input = Input(shape=(None, config['preprocessing_params']['max_word_length'],), name='char_input')
#             if features['char_embeddings'].weights():
#                 embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].shape()[0],
#                                                            output_dim=features['char_embeddings'].shape()[0],
#                                                            weights=[features['char_embeddings'].weights()],
#                                                            trainable=False))(character_input)
#             else:
#                 embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].vocab_length(),
#                                                            output_dim=30,
#                                                            embeddings_initializer=RandomUniform(minval=-0.5,
#                                                                                                 maxval=0.5)),
#                                                  name='char_embedding')(character_input)

#             conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=128,
#                                                 padding='same', activation='tanh',
#                                                 strides=1, name='conv1d_out'))(
#                 embed_char_out)
#             maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
#             char = TimeDistributed(Flatten())(maxpool_out)
#             char = Dropout(0.5)(char)
#             output = concatenate([words, case, char])
#             output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
#             output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
#             output = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
            
# #             if config['joint_relation']:
# #                 relation = Dense(features['relation_embeddings'].shape()[0], activation='softmax')(output)

#             output = Dense(features['label_embeddings'].shape()[0], activation='softmax')(output)

#             if config['use_crf']:
#                 crf = CRF(features['label_embeddings'].shape()[0], sparse_target=True)
#                 output = crf(output)
#                 model = Model(inputs=[words_input, character_input, casing_input], outputs=[output])
#                 model.compile(optimizer='adam', loss=crf.loss_function)
#             else:
#                 model = Model(inputs=[words_input, character_input, casing_input], outputs=[output])
#                 model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")

#             return model
        
       
        
#         if model_name == 'lstm_cnn':
#             words_input = Input(shape=(None,), dtype='int32', name='words_input')
#             words = Embedding(input_dim=features['word_embeddings'].shape()[0],
#                                 output_dim=features['word_embeddings'].shape()[1],
#                                 weights=[features['word_embeddings'].weights()],
#                                 trainable=False,
#                                 name='words_embedding')(words_input)

#             casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
#             case = Embedding(input_dim=features['case_embeddings'].shape()[0],
#                                  output_dim=features['case_embeddings'].shape()[0],
#                                  weights=[features['case_embeddings'].weights()],
#                                  trainable=False,
#                                  name='case_embedding')(casing_input)

#             character_input = Input(shape=(None, config['preprocessing_params']['max_word_length'],), name='char_input')
#             if features['char_embeddings'].weights():
#                 embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].shape()[0],
#                                                                output_dim=features['char_embeddings'].shape()[0],
#                                                                weights=[features['char_embeddings'].weights()],
#                                                                trainable=False))(character_input)
#             else:
#                 embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].vocab_length(),
#                                                                output_dim=30,
#                                                                embeddings_initializer=RandomUniform(minval=-0.5,
#                                                                                                     maxval=0.5)),
#                                                      name='char_embedding')(character_input)

#             conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=128,
#                                                     padding='same', activation='tanh',
#                                                     strides=1, name='conv1d_out'))(
#                     embed_char_out)
#             maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
#             char = TimeDistributed(Flatten())(maxpool_out)
#             char = Dropout(0.5)(char)
#             output = concatenate([words, case, char])
#             output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
#             lstm_last = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)

#             if config['joint_relation']:
#                 ner = Dense(features['label_embeddings'].shape()[0], activation='softmax')(lstm_last)
#                 in_relation = concatenate([ner, lstm_last])
#                 relation = Dense(features['relation_embeddings'].shape()[0], activation='softmax')(in_relation)

#             model = Model(inputs=[words_input, character_input, casing_input], outputs=[ner, relation])
#             model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")

#             return model
     
        if model_name == 'lstm_cnn':
            words_input = Input(shape=(None,), dtype='int32', name='words_input')
            words = Embedding(input_dim=features['word_embeddings'].shape()[0],
                              output_dim=features['word_embeddings'].shape()[1],
                              weights=[features['word_embeddings'].weights()],
                              trainable=False,
                              name='words_embedding')(words_input)

            casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
            case = Embedding(input_dim=features['case_embeddings'].shape()[0],
                             output_dim=features['case_embeddings'].shape()[0],
                             weights=[features['case_embeddings'].weights()],
                             trainable=False,
                             name='case_embedding')(casing_input)

            character_input = Input(shape=(None, config['preprocessing_params']['max_word_length'],), name='char_input')
            if features['char_embeddings'].weights():
                embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].shape()[0],
                                                           output_dim=features['char_embeddings'].shape()[0],
                                                           weights=[features['char_embeddings'].weights()],
                                                           trainable=False))(character_input)
            else:
                embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].vocab_length(),
                                                           output_dim=30,
                                                           embeddings_initializer=RandomUniform(minval=-0.5,
                                                                                                maxval=0.5)),
                                                 name='char_embedding')(character_input)

            conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=128,
                                                padding='same', activation='tanh',
                                                strides=1, name='conv1d_out'))(
                embed_char_out)
            maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
            char = TimeDistributed(Flatten())(maxpool_out)
            char = Dropout(0.5)(char)
            output = concatenate([words, case, char])
            output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
            output = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(output)
            
#             if config['joint_relation']:
#                 relation = Dense(features['relation_embeddings'].shape()[0], activation='softmax')(output)

            output = Dense(features['label_embeddings'].shape()[0], activation='softmax')(output)

            if config['use_crf']:
                crf = CRF(features['label_embeddings'].shape()[0], sparse_target=True)
                output = crf(output)
                model = Model(inputs=[words_input, character_input, casing_input], outputs=[output])
                model.compile(optimizer='adam', loss=crf.loss_function)
            else:
                model = Model(inputs=[words_input, character_input, casing_input], outputs=[output])
                model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")

            return model
        
        
        """  
        ###### Models below will be used later... ########
        if model_name == 'flair':
            words_input = Input(shape=(None,), dtype='int32', name='words_input')
            words = Embedding(input_dim=features['word_embeddings'].shape()[0],
                              output_dim=features['word_embeddings'].shape()[1],
                              weights=[features['word_embeddings'].weights()],
                              trainable=False,
                              name='words_embedding')(words_input)

            flair_input = Input(shape=(None, features['flair_embeddings'].shape()[0],),
                                dtype="float32",
                                name="flair_input")

            output = concatenate([words, flair_input])
            output = Bidirectional(LSTM(256, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

            output = TimeDistributed(Dense(features['label_embeddings'].shape()[0],
                                           activation='tanh'))(output)

            if use_crf:
                crf = CRF(features['label_embeddings'].shape()[0], sparse_target=True)
                output = crf(output)
                model = Model(inputs=[words_input, flair_input], outputs=[output])
                model.compile(optimizer='nadam', loss=crf.loss_function)
            else:
                model = Model(inputs=[words_input, flair_input], outputs=[output])
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

            return model
        
        if model_name == 'elmo':
            words_input = Input(shape=(None,), dtype='int32', name='words_input')
            words = Embedding(input_dim=features['word_embeddings'].shape()[0],
                              output_dim=features['word_embeddings'].shape()[1],
                              weights=[features['word_embeddings'].weights()],
                              trainable=False)(words_input)

            text_input = Input(shape=(None, features['elmo_embeddings'].shape()[0],
                                      features['elmo_embeddings'].shape()[1],),
                               dtype='float32', name='text_input')
            
            linear_elmo = TimeDistributed(Dense(1,
                                                activation='linear',
                                                input_shape=(features['elmo_embeddings'].shape()[0],
                                                             features['elmo_embeddings'].shape()[1])))(text_input)
            linear_elmo = TimeDistributed(Flatten())(linear_elmo)

            casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
            case = Embedding(input_dim=features['case_embeddings'].shape()[0],
                             output_dim=features['case_embeddings'].shape()[0],
                             weights=[features['case_embeddings'].weights()],
                             trainable=False)(casing_input)

            character_input = Input(shape=(None, config['max_word_length'],), name='char_input')
            if features['char_embeddings'].weights():
                embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].shape()[0],
                                                           output_dim=features['char_embeddings'].shape()[0],
                                                           weights=[features['char_embeddings'].weights()],
                                                           trainable=False))(character_input)
            else:
                embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].vocab_length(),
                                                           output_dim=30,
                                                           mask_zero=True,
                                                           embeddings_initializer=RandomUniform(minval=-0.5,
                                                                                                maxval=0.5)),
                                                 name='char_embedding')(character_input)

            conv1d_out = TimeDistributed(Conv1D(kernel_size=3,
                                                filters=128,
                                                padding='same',
                                                activation='tanh',
                                                strides=1))(
                embed_char_out)
            maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
            char = TimeDistributed(Flatten())(maxpool_out)
            char = Dropout(0.5)(char)
            output = concatenate([words, case, char, linear_elmo])
            output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
            output = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

            output = TimeDistributed(Dense(features['label_embeddings'].shape()[0],
                                           activation='tanh'))(output)

            if use_crf:
                crf = CRF(features['label_embeddings'].shape()[0], sparse_target=True)
                output = crf(output)
                model = Model(inputs=[words_input, character_input, casing_input, text_input], outputs=[output])
                model.compile(optimizer='adam', loss=crf.loss_function)
            else:
                model = Model(inputs=[words_input, character_input, casing_input, text_input], outputs=[output])
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

            return model
        
        if model_name == 'elmo_flair':
            words_input = Input(shape=(None,), dtype='int32', name='words_input')
            words = Embedding(input_dim=features['word_embeddings'].shape()[0],
                              output_dim=features['word_embeddings'].shape()[1],
                              weights=[features['word_embeddings'].weights()],
                              trainable=False,
                              name='words_embedding')(words_input)

            casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
            case = Embedding(input_dim=features['case_embeddings'].shape()[0],
                             output_dim=features['case_embeddings'].shape()[0],
                             weights=[features['case_embeddings'].weights()],
                             trainable=False,
                             name='case_embedding')(casing_input)

            text_input = Input(shape=(None, features['elmo_embeddings'].shape()[0],
                                      features['elmo_embeddings'].shape()[1],),
                               dtype='float32', name='text_input')

            linear_elmo = TimeDistributed(Dense(1,
                                                activation='linear',
                                                input_shape=(features['elmo_embeddings'].shape()[0],
                                                             features['elmo_embeddings'].shape()[1])))(text_input)
            linear_elmo = TimeDistributed(Flatten())(linear_elmo)

            flair_input = Input(shape=(None, features['flair_embeddings'].shape()[0],), dtype="float32",
                                name="flair_input")
            character_input = Input(shape=(None, config['max_word_length'],), name='char_input')
            if features['char_embeddings'].weights():
                embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].shape()[0],
                                                           output_dim=features['char_embeddings'].shape()[0],
                                                           weights=[features['char_embeddings'].weights()],
                                                           trainable=False))(character_input)
            else:
                embed_char_out = TimeDistributed(Embedding(input_dim=features['char_embeddings'].vocab_length(),
                                                           output_dim=30,
                                                           embeddings_initializer=RandomUniform(minval=-0.5,
                                                                                                maxval=0.5)),
                                                 name='char_embedding')(character_input)

            conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=128,
                                                padding='same', activation='tanh',
                                                strides=1, name='conv1d_out'))(
                embed_char_out)
            maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
            char = TimeDistributed(Flatten())(maxpool_out)
            char = Dropout(0.5)(char)
            output = concatenate([words, case, char, linear_elmo, flair_input])

            output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
            output = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

            output = TimeDistributed(Dense(features['label_embeddings'].shape()[0],
                                           activation='softmax'))(output)

            if use_crf:
                crf = CRF(features['label_embeddings'].shape()[0], sparse_target=True)
                output = crf(output)
                model = Model(inputs=[words_input, character_input, casing_input, text_input, flair_input],
                              outputs=[output])
                model.compile(optimizer='nadam', loss=crf.loss_function)
            else:
                model = Model(inputs=[words_input, character_input, casing_input, text_input, flair_input],
                              outputs=[output])
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

            return model
        """
