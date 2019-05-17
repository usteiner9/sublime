from __future__ import absolute_import
from __future__ import division
import keras.backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy
from keras.constraints import Constraint

"""
Author: Philipp Gross, https://github.com/phipleg/keras/blob/crf/keras/layers/crf.py
"""

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec


def path_energy(y, x, U, b_start=None, b_end=None, mask=None):
    """Calculates the energy of a tag path y for a given input x (with mask),
    transition energies U and boundary energies b_start, b_end."""
    x = add_boundary_energy(x, b_start, b_end, mask)
    return path_energy0(y, x, U, mask)


def path_energy0(y, x, U, mask=None):
    """Path energy without boundary potential handling."""
    n_classes = K.shape(x)[2]
    y_one_hot = K.one_hot(y, n_classes)

    # Tag path energy
    energy = K.sum(x * y_one_hot, 2)
    energy = K.sum(energy, 1)

    # Transition energy
    y_t = y[:, :-1]
    y_tp1 = y[:, 1:]
    U_flat = K.reshape(U, [-1])
    # Convert 2-dim indices (y_t, y_tp1) of U to 1-dim indices of U_flat:
    flat_indices = y_t * n_classes + y_tp1
    U_y_t_tp1 = K.gather(U_flat, flat_indices)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        y_t_mask = mask[:, :-1]
        y_tp1_mask = mask[:, 1:]
        U_y_t_tp1 *= y_t_mask * y_tp1_mask

    energy += K.sum(U_y_t_tp1, axis=1)

    return energy


def sparse_chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    """Given the true sparsely encoded tag sequence y, input x (with mask),
    transition energies U, boundary energies b_start and b_end, it computes
    the loss function of a Linear Chain Conditional Random Field:
    loss(y, x) = NNL(P(y|x)), where P(y|x) = exp(E(y, x)) / Z.
    So, loss(y, x) = - E(y, x) + log(Z)
    Here, E(y, x) is the tag path energy, and Z is the normalization constant.
    The values log(Z) is also called free energy.
    """
    x = add_boundary_energy(x, b_start, b_end, mask)
    energy = path_energy0(y, x, U, mask)
    energy -= free_energy0(x, U, mask)
    return K.expand_dims(-energy, -1)


def chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    """Variant of sparse_chain_crf_loss but with one-hot encoded tags y."""
    y_sparse = K.argmax(y, -1)
    y_sparse = K.cast(y_sparse, 'int32')
    return sparse_chain_crf_loss(y_sparse, x, U, b_start, b_end, mask)


def add_boundary_energy(x, b_start=None, b_end=None, mask=None):
    """Given the observations x, it adds the start boundary energy b_start (resp.
    end boundary energy b_end on the start (resp. end) elements and multiplies
    the mask."""
    if mask is None:
        if b_start is not None:
            x = K.concatenate([x[:, :1, :] + b_start, x[:, 1:, :]], axis=1)
        if b_end is not None:
            x = K.concatenate([x[:, :-1, :], x[:, -1:, :] + b_end], axis=1)
    else:
        mask = K.cast(mask, K.floatx())
        mask = K.expand_dims(mask, 2)
        x *= mask
        if b_start is not None:
            mask_r = K.concatenate([K.zeros_like(mask[:, :1]), mask[:, :-1]], axis=1)
            start_mask = K.cast(K.greater(mask, mask_r), K.floatx())
            x = x + start_mask * b_start
        if b_end is not None:
            mask_l = K.concatenate([mask[:, 1:], K.zeros_like(mask[:, -1:])], axis=1)
            end_mask = K.cast(K.greater(mask, mask_l), K.floatx())
            x = x + end_mask * b_end
    return x


def viterbi_decode(x, U, b_start=None, b_end=None, mask=None):
    """Computes the best tag sequence y for a given input x, i.e. the one that
    maximizes the value of path_energy."""
    x = add_boundary_energy(x, b_start, b_end, mask)

    alpha_0 = x[:, 0, :]
    gamma_0 = K.zeros_like(alpha_0)
    initial_states = [gamma_0, alpha_0]
    _, gamma = _forward(x,
                        lambda B: [K.cast(K.argmax(B, axis=1), K.floatx()), K.max(B, axis=1)],
                        initial_states,
                        U,
                        mask)
    y = _backward(gamma, mask)
    return y


def free_energy(x, U, b_start=None, b_end=None, mask=None):
    """Computes efficiently the sum of all path energies for input x, when
    runs over all possible tag sequences."""
    x = add_boundary_energy(x, b_start, b_end, mask)
    return free_energy0(x, U, mask)


def free_energy0(x, U, mask=None):
    """Free energy without boundary potential handling."""
    initial_states = [x[:, 0, :]]
    last_alpha, _ = _forward(x,
                             lambda B: [K.logsumexp(B, axis=1)],
                             initial_states,
                             U,
                             mask)
    return last_alpha[:, 0]


def _forward(x, reduce_step, initial_states, U, mask=None):
    """Forward recurrence of the linear chain crf."""

    def _forward_step(energy_matrix_t, states):
        alpha_tm1 = states[-1]
        new_states = reduce_step(K.expand_dims(alpha_tm1, 2) + energy_matrix_t)
        return new_states[0], new_states

    U_shared = K.expand_dims(K.expand_dims(U, 0), 0)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        mask_U = K.expand_dims(K.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
        U_shared = U_shared * mask_U

    inputs = K.expand_dims(x[:, 1:, :], 2) + U_shared
    inputs = K.concatenate([inputs, K.zeros_like(inputs[:, -1:, :, :])], axis=1)

    last, values, _ = K.rnn(_forward_step, inputs, initial_states)
    return last, values


def batch_gather(reference, indices):
    ref_shape = K.shape(reference)
    batch_size = ref_shape[0]
    n_classes = ref_shape[1]
    flat_indices = K.arange(0, batch_size) * n_classes + K.flatten(indices)
    return K.gather(K.flatten(reference), flat_indices)


def _backward(gamma, mask):
    """Backward recurrence of the linear chain crf."""
    gamma = K.cast(gamma, 'int32')

    def _backward_step(gamma_t, states):
        y_tm1 = K.squeeze(states[0], 0)
        y_t = batch_gather(gamma_t, y_tm1)
        return y_t, [K.expand_dims(y_t, 0)]

    initial_states = [K.expand_dims(K.zeros_like(gamma[:, 0, 0]), 0)]
    _, y_rev, _ = K.rnn(_backward_step,
                        gamma,
                        initial_states,
                        go_backwards=True)
    y = K.reverse(y_rev, 1)

    if mask is not None:
        mask = K.cast(mask, dtype='int32')
        # mask output
        y *= mask
        # set masked values to -1
        y += -(1 - mask)
    return y


class ChainCRF(Layer):
    """A Linear Chain Conditional Random Field output layer.
    It carries the loss function and its weights for computing
    the global tag sequence scores. While training it acts as
    the identity function that passes the inputs to the subsequently
    used loss function. While testing it applies Viterbi decoding
    and returns the best scoring tag sequence as one-hot encoded vectors.
    # Arguments
        init: weight initialization function for chain energies U.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializers](../initializers.md)).
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the transition weight matrix.
        b_start_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the start bias b.
        b_end_regularizer: instance of [WeightRegularizer](../regularizers.md)
            module, applied to the end bias b.
        b_start_constraint: instance of the [constraints](../constraints.md)
            module, applied to the start bias b.
        b_end_constraint: instance of the [constraints](../constraints.md)
            module, applied to the end bias b.
        weights: list of Numpy arrays for initializing [U, b_start, b_end].
            Thus it should be a list of 3 elements of shape
            [(n_classes, n_classes), (n_classes, ), (n_classes, )]
    # Input shape
        3D tensor with shape `(nb_samples, timesteps, nb_classes)`, where
        ´timesteps >= 2`and `nb_classes >= 2`.
    # Output shape
        Same shape as input.
    # Masking
        This layer supports masking for input sequences of variable length.
    # Example
    ```python
    # As the last layer of sequential layer with
    # model.output_shape == (None, timesteps, nb_classes)
    crf = ChainCRF()
    model.add(crf)
    # now: model.output_shape == (None, timesteps, nb_classes)
    # Compile model with chain crf loss (and one-hot encoded labels) and accuracy
    model.compile(loss=crf.loss, optimizer='sgd', metrics=['accuracy'])
    # Alternatively, compile model with sparsely encoded labels and sparse accuracy:
    model.compile(loss=crf.sparse_loss, optimizer='sgd', metrics=['sparse_categorical_accuracy'])
    ```
    # Gotchas
    ## Model loading
    When you want to load a saved model that has a crf output, then loading
    the model with 'keras.models.load_model' won't work properly because
    the reference of the loss function to the transition parameters is lost. To
    fix this, you need to use the parameter 'custom_objects' as follows:
    ```python
    from keras.layer.crf import create_custom_objects:
    model = keras.models.load_model(filename, custom_objects=create_custom_objects())
    ```
    ## Temporal sample weights
    Given a ChainCRF instance crf both loss functions, crf.loss and crf.sparse_loss
    return a tensor of shape (batch_size, 1) and not (batch_size, maxlen).
    that sample weighting in temporal mode.
    """
    def __init__(self, init='glorot_uniform',
                 U_regularizer=None,
                 b_start_regularizer=None,
                 b_end_regularizer=None,
                 U_constraint=None,
                 b_start_constraint=None,
                 b_end_constraint=None,
                 weights=None,
                 **kwargs):
        super(ChainCRF, self).__init__(**kwargs)
        self.init = initializers.get(init)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_start_regularizer = regularizers.get(b_start_regularizer)
        self.b_end_regularizer = regularizers.get(b_end_regularizer)
        self.U_constraint = constraints.get(U_constraint)
        self.b_start_constraint = constraints.get(b_start_constraint)
        self.b_end_constraint = constraints.get(b_end_constraint)

        self.initial_weights = weights

        self.supports_masking = True
        self.uses_learning_phase = True
        self.input_spec = [InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1], input_shape[2])

    def compute_mask(self, input, mask=None):
        if mask is not None:
            return K.any(mask, axis=1)
        return mask

    def _fetch_mask(self):
        mask = None
        if self._inbound_nodes:
            mask = self._inbound_nodes[0].input_masks[0]
        return mask

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_steps is None or n_steps >= 2
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, n_steps, n_classes))]

        self.U = self.add_weight((n_classes, n_classes),
                                 initializer=self.init,
                                 name='U',
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)

        self.b_start = self.add_weight((n_classes, ),
                                       initializer='zero',
                                       name='b_start',
                                       regularizer=self.b_start_regularizer,
                                       constraint=self.b_start_constraint)

        self.b_end = self.add_weight((n_classes, ),
                                     initializer='zero',
                                     name='b_end',
                                     regularizer=self.b_end_regularizer,
                                     constraint=self.b_end_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        y_pred = viterbi_decode(x, self.U, self.b_start, self.b_end, mask)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)
        return K.in_train_phase(x, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        """Linear Chain Conditional Random Field loss function.
        """
        mask = self._fetch_mask()
        return chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def sparse_loss(self, y_true, y_pred):
        """Linear Chain Conditional Random Field loss function with sparse
        tag sequences.
        """
        y_true = K.cast(y_true, 'int32')
        y_true = K.squeeze(y_true, 2)
        mask = self._fetch_mask()
        return sparse_chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def get_config(self):
        config = {
            'init': initializers.serialize(self.init),
            'U_regularizer': regularizers.serialize(self.U_regularizer),
            'b_start_regularizer': regularizers.serialize(self.b_start_regularizer),
            'b_end_regularizer': regularizers.serialize(self.b_end_regularizer),
            'U_constraint': constraints.serialize(self.U_constraint),
            'b_start_constraint': constraints.serialize(self.b_start_constraint),
            'b_end_constraint': constraints.serialize(self.b_end_constraint)
        }
        base_config = super(ChainCRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftmaxConstraint(Constraint):
    """Constrains the weights to be non-negative.
    """

    def __call__(self, w):
        den = K.sum(K.exp(K.cast(w, "float32")), axis=0)
        w = w / den
        return w


class CRF(Layer):
    """An implementation of linear chain conditional random field (CRF).

    An linear chain CRF is defined to maximize the following likelihood function:

    $$ L(W, U, b; y_1, ..., y_n) := \frac{1}{Z} \sum_{y_1, ..., y_n} \exp(-a_1' y_1 - a_n' y_n
        - \sum_{k=1^n}((f(x_k' W + b) y_k) + y_1' U y_2)), $$

    where:
        $Z$: normalization constant
        $x_k, y_k$:  inputs and outputs

    This implementation has two modes for optimization:
    1. (`join mode`) optimized by maximizing join likelihood, which is optimal in theory of statistics.
       Note that in this case, CRF must be the output/last layer.
    2. (`marginal mode`) return marginal probabilities on each time step and optimized via composition
       likelihood (product of marginal likelihood), i.e., using `categorical_crossentropy` loss.
       Note that in this case, CRF can be either the last layer or an intermediate layer (though not explored).

    For prediction (test phrase), one can choose either Viterbi best path (class indices) or marginal
    probabilities if probabilities are needed. However, if one chooses *join mode* for training,
    Viterbi output is typically better than marginal output, but the marginal output will still perform
    reasonably close, while if *marginal mode* is used for training, marginal output usually performs
    much better. The default behavior is set according to this observation.

    In addition, this implementation supports masking and accepts either onehot or sparse target.


    # Examples

    ```python
        model = Sequential()
        model.add(Embedding(3001, 300, mask_zero=True)(X)

        # use learn_mode = 'join', test_mode = 'viterbi', sparse_target = True (label indice output)
        crf = CRF(10, sparse_target=True)
        model.add(crf)

        # crf.accuracy is default to Viterbi acc if using join-mode (default).
        # One can add crf.marginal_acc if interested, but may slow down learning
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

        # y must be label indices (with shape 1 at dim 3) here, since `sparse_target=True`
        model.fit(x, y)

        # prediction give onehot representation of Viterbi best path
        y_hat = model.predict(x_test)
    ```


    # Arguments
        units: Positive integer, dimensionality of the output space.
        learn_mode: Either 'join' or 'marginal'.
            The former train the model by maximizing join likelihood while the latter
            maximize the product of marginal likelihood over all time steps.
        test_mode: Either 'viterbi' or 'marginal'.
            The former is recommended and as default when `learn_mode = 'join'` and
            gives one-hot representation of the best path at test (prediction) time,
            while the latter is recommended and chosen as default when `learn_mode = 'marginal'`,
            which produces marginal probabilities for each time step.
        sparse_target: Boolean (default False) indicating if provided labels are one-hot or
            indices (with shape 1 at dim 3).
        use_boundary: Boolean (default True) indicating if trainable start-end chain energies
            should be added to model.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        chain_initializer: Initializer for the `chain_kernel` weights matrix,
            used for the CRF chain energy.
            (see [initializers](../initializers.md)).
        boundary_initializer: Initializer for the `left_boundary`, 'right_boundary' weights vectors,
            used for the start/left and end/right boundary energy.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        chain_regularizer: Regularizer function applied to
            the `chain_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        boundary_regularizer: Regularizer function applied to
            the 'left_boundary', 'right_boundary' weight vectors
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        chain_constraint: Constraint function applied to
            the `chain_kernel` weights matrix
            (see [constraints](../constraints.md)).
        boundary_constraint: Constraint function applied to
            the `left_boundary`, `right_boundary` weights vectors
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_samples, timesteps, units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    """

    def __init__(self, units,
                 learn_mode='join',
                 test_mode=None,
                 sparse_target=False,
                 use_boundary=True,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 chain_initializer='orthogonal',
                 bias_initializer='zeros',
                 boundary_initializer='zeros',
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 boundary_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 boundary_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 unroll=False,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.learn_mode = learn_mode
        assert self.learn_mode in ['join', 'marginal']
        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = 'viterbi' if self.learn_mode == 'join' else 'marginal'
        else:
            assert self.test_mode in ['viterbi', 'marginal']
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.boundary_initializer = initializers.get(boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.unroll = unroll

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight((self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.chain_kernel = self.add_weight((self.units, self.units),
                                            name='chain_kernel',
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_boundary:
            self.left_boundary = self.add_weight((self.units,),
                                                 name='left_boundary',
                                                 initializer=self.boundary_initializer,
                                                 regularizer=self.boundary_regularizer,
                                                 constraint=self.boundary_constraint)
            self.right_boundary = self.add_weight((self.units,),
                                                  name='right_boundary',
                                                  initializer=self.boundary_initializer,
                                                  regularizer=self.boundary_regularizer,
                                                  constraint=self.boundary_constraint)
        self.built = True

    def call(self, X, mask=None):
        if mask is not None:
            assert K.ndim(mask) == 2, 'Input mask to CRF must have dim 2 if not None'

        if self.test_mode == 'viterbi':
            test_output = self.viterbi_decoding(X, mask)
        else:
            test_output = self.get_marginal_prob(X, mask)

        self.uses_learning_phase = True
        if self.learn_mode == 'join':
            train_output = K.zeros_like(K.dot(X, self.kernel))
            out = K.in_train_phase(train_output, test_output)
        else:
            if self.test_mode == 'viterbi':
                train_output = self.get_marginal_prob(X, mask)
                out = K.in_train_phase(train_output, test_output)
            else:
                out = test_output
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.units,)

    def compute_mask(self, input, mask=None):
        if mask is not None and self.learn_mode == 'join':
            return K.any(mask, axis=1)
        return mask

    def get_config(self):
        config = {'units': self.units,
                  'learn_mode': self.learn_mode,
                  'test_mode': self.test_mode,
                  'use_boundary': self.use_boundary,
                  'use_bias': self.use_bias,
                  'sparse_target': self.sparse_target,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'chain_initializer': initializers.serialize(self.chain_initializer),
                  'boundary_initializer': initializers.serialize(self.boundary_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'activation': activations.serialize(self.activation),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'chain_regularizer': regularizers.serialize(self.chain_regularizer),
                  'boundary_regularizer': regularizers.serialize(self.boundary_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'chain_constraint': constraints.serialize(self.chain_constraint),
                  'boundary_constraint': constraints.serialize(self.boundary_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'unroll': self.unroll}
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def loss_function(self):
        if self.learn_mode == 'join':
            def loss(y_true, y_pred):
                assert self._inbound_nodes, 'CRF has not connected to any layer.'
                assert not self._outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
                if self.sparse_target:
                    y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), self.units)
                X = self._inbound_nodes[0].input_tensors[0]
                mask = self._inbound_nodes[0].input_masks[0]
                nloglik = self.get_negative_log_likelihood(y_true, X, mask)
                return nloglik
            return loss
        else:
            if self.sparse_target:
                return sparse_categorical_crossentropy
            else:
                return categorical_crossentropy

    @property
    def accuracy(self):
        if self.test_mode == 'viterbi':
            return self.viterbi_acc
        else:
            return self.marginal_acc

    @staticmethod
    def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        if mask is None:
            return K.mean(judge)
        else:
            mask = K.cast(mask, K.floatx())
            return K.sum(judge * mask) / K.sum(mask)

    @property
    def viterbi_acc(self):
        def acc(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.viterbi_decoding(X, mask)
            return self._get_accuracy(y_true, y_pred, mask, self.sparse_target)
        acc.func_name = 'viterbi_acc'
        return acc

    @property
    def marginal_acc(self):
        def acc(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.get_marginal_prob(X, mask)
            return self._get_accuracy(y_true, y_pred, mask, self.sparse_target)
        acc.func_name = 'marginal_acc'
        return acc

    @staticmethod
    def softmaxNd(x, axis=-1):
        m = K.max(x, axis=axis, keepdims=True)
        exp_x = K.exp(x - m)
        prob_x = exp_x / K.sum(exp_x, axis=axis, keepdims=True)
        return prob_x

    @staticmethod
    def shift_left(x, offset=1):
        assert offset > 0
        return K.concatenate([x[:, offset:], K.zeros_like(x[:, :offset])], axis=1)

    @staticmethod
    def shift_right(x, offset=1):
        assert offset > 0
        return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)

    def add_boundary_energy(self, energy, mask, start, end):
        start = K.expand_dims(K.expand_dims(start, 0), 0)
        end = K.expand_dims(K.expand_dims(end, 0), 0)
        if mask is None:
            energy = K.concatenate([energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)
            energy = K.concatenate([energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            mask = K.expand_dims(K.cast(mask, K.floatx()))
            start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())
            end_mask = K.cast(K.greater(self.shift_left(mask), mask), K.floatx())
            energy = energy + start_mask * start
            energy = energy + end_mask * end
        return energy

    def get_log_normalization_constant(self, input_energy, mask, **kwargs):
        """Compute logarithm of the normalization constant Z, where
        Z = sum exp(-E) -> logZ = log sum exp(-E) =: -nlogZ
        """
        # should have logZ[:, i] == logZ[:, j] for any i, j
        logZ = self.recursion(input_energy, mask, return_sequences=False, **kwargs)
        return logZ[:, 0]

    def get_energy(self, y_true, input_energy, mask):
        """Energy = a1' y1 + u1' y1 + y1' U y2 + u2' y2 + y2' U y3 + u3' y3 + an' y3
        """
        input_energy = K.sum(input_energy * y_true, 2)  # (B, T)
        chain_energy = K.sum(K.dot(y_true[:, :-1, :], self.chain_kernel) * y_true[:, 1:, :], 2)  # (B, T-1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            chain_mask = mask[:, :-1] * mask[:, 1:]  # (B, T-1), mask[:,:-1]*mask[:,1:] makes it work with any padding
            input_energy = input_energy * mask
            chain_energy = chain_energy * chain_mask
        total_energy = K.sum(input_energy, -1) + K.sum(chain_energy, -1)  # (B, )

        return total_energy

    def get_negative_log_likelihood(self, y_true, X, mask):
        """Compute the loss, i.e., negative log likelihood (normalize by number of time steps)
           likelihood = 1/Z * exp(-E) ->  neg_log_like = - log(1/Z * exp(-E)) = logZ + E
        """
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
        energy = self.get_energy(y_true, input_energy, mask)
        logZ = self.get_log_normalization_constant(input_energy, mask, input_length=K.int_shape(X)[1])
        nloglik = logZ + energy
        if mask is not None:
            nloglik = nloglik / K.sum(K.cast(mask, K.floatx()), 1)
        else:
            nloglik = nloglik / K.cast(K.shape(X)[1], K.floatx())
        return nloglik

    def step(self, input_energy_t, states, return_logZ=True):
        # not in the following  `prev_target_val` has shape = (B, F)
        # where B = batch_size, F = output feature dim
        # Note: `i` is of float32, due to the behavior of `K.rnn`
        prev_target_val, i, chain_energy = states[:3]
        t = K.cast(i[0, 0], dtype='int32')
        if len(states) > 3:
            if K.backend() == 'theano':
                m = states[3][:, t:(t + 2)]
            else:
                m = K.tf.slice(states[3], [0, t], [-1, 2])
            input_energy_t = input_energy_t * K.expand_dims(m[:, 0])
            chain_energy = chain_energy * K.expand_dims(K.expand_dims(m[:, 0] * m[:, 1]))  # (1, F, F)*(B, 1, 1) -> (B, F, F)
        if return_logZ:
            energy = chain_energy + K.expand_dims(input_energy_t - prev_target_val, 2)  # shapes: (1, B, F) + (B, F, 1) -> (B, F, F)
            new_target_val = K.logsumexp(-energy, 1)  # shapes: (B, F)
            return new_target_val, [new_target_val, i + 1]
        else:
            energy = chain_energy + K.expand_dims(input_energy_t + prev_target_val, 2)
            min_energy = K.min(energy, 1)
            argmin_table = K.cast(K.argmin(energy, 1), K.floatx())  # cast for tf-version `K.rnn`
            return argmin_table, [min_energy, i + 1]

    def recursion(self, input_energy, mask=None, go_backwards=False, return_sequences=True, return_logZ=True, input_length=None):
        """Forward (alpha) or backward (beta) recursion

        If `return_logZ = True`, compute the logZ, the normalization constant:

        \[ Z = \sum_{y1, y2, y3} exp(-E) # energy
          = \sum_{y1, y2, y3} exp(-(u1' y1 + y1' W y2 + u2' y2 + y2' W y3 + u3' y3))
          = sum_{y2, y3} (exp(-(u2' y2 + y2' W y3 + u3' y3)) sum_{y1} exp(-(u1' y1' + y1' W y2))) \]

        Denote:
            \[ S(y2) := sum_{y1} exp(-(u1' y1 + y1' W y2)), \]
            \[ Z = sum_{y2, y3} exp(log S(y2) - (u2' y2 + y2' W y3 + u3' y3)) \]
            \[ logS(y2) = log S(y2) = log_sum_exp(-(u1' y1' + y1' W y2)) \]
        Note that:
              yi's are one-hot vectors
              u1, u3: boundary energies have been merged

        If `return_logZ = False`, compute the Viterbi's best path lookup table.
        """
        chain_energy = self.chain_kernel
        chain_energy = K.expand_dims(chain_energy, 0)  # shape=(1, F, F): F=num of output features. 1st F is for t-1, 2nd F for t
        prev_target_val = K.zeros_like(input_energy[:, 0, :])  # shape=(B, F), dtype=float32

        if go_backwards:
            input_energy = K.reverse(input_energy, 1)
            if mask is not None:
                mask = K.reverse(mask, 1)

        initial_states = [prev_target_val, K.zeros_like(prev_target_val[:, :1])]
        constants = [chain_energy]

        if mask is not None:
            mask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1), K.floatx())
            constants.append(mask2)

        def _step(input_energy_i, states):
            return self.step(input_energy_i, states, return_logZ)

        target_val_last, target_val_seq, _ = K.rnn(_step, input_energy, initial_states, constants=constants,
                                                   input_length=input_length, unroll=self.unroll)

        if return_sequences:
            if go_backwards:
                target_val_seq = K.reverse(target_val_seq, 1)
            return target_val_seq
        else:
            return target_val_last

    def forward_recursion(self, input_energy, **kwargs):
        return self.recursion(input_energy, **kwargs)

    def backward_recursion(self, input_energy, **kwargs):
        return self.recursion(input_energy, go_backwards=True, **kwargs)

    def get_marginal_prob(self, X, mask=None):
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
        input_length = K.int_shape(X)[1]
        alpha = self.forward_recursion(input_energy, mask=mask, input_length=input_length)
        beta = self.backward_recursion(input_energy, mask=mask, input_length=input_length)
        if mask is not None:
            input_energy = input_energy * K.expand_dims(K.cast(mask, K.floatx()))
        margin = -(self.shift_right(alpha) + input_energy + self.shift_left(beta))
        return self.softmaxNd(margin)

    def viterbi_decoding(self, X, mask=None):
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)

        argmin_tables = self.recursion(input_energy, mask, return_logZ=False)
        argmin_tables = K.cast(argmin_tables, 'int32')

        # backward to find best path, `initial_best_idx` can be any, as all elements in the last argmin_table are the same
        argmin_tables = K.reverse(argmin_tables, 1)
        initial_best_idx = [K.expand_dims(argmin_tables[:, 0, 0])]  # matrix instead of vector is required by tf `K.rnn`
        if K.backend() == 'theano':
            initial_best_idx = [K.T.unbroadcast(initial_best_idx[0], 1)]

        def gather_each_row(params, indices):
            n = K.shape(indices)[0]
            if K.backend() == 'theano':
                return params[K.T.arange(n), indices]
            else:
                indices = K.transpose(K.stack([K.tf.range(n), indices]))
                return K.tf.gather_nd(params, indices)

        def find_path(argmin_table, best_idx):
            next_best_idx = gather_each_row(argmin_table, best_idx[0][:, 0])
            next_best_idx = K.expand_dims(next_best_idx)
            if K.backend() == 'theano':
                next_best_idx = K.T.unbroadcast(next_best_idx, 1)
            return next_best_idx, [next_best_idx]

        _, best_paths, _ = K.rnn(find_path, argmin_tables, initial_best_idx, input_length=K.int_shape(X)[1], unroll=self.unroll)
        best_paths = K.reverse(best_paths, 1)
        best_paths = K.squeeze(best_paths, 2)

        return K.one_hot(best_paths, self.units)