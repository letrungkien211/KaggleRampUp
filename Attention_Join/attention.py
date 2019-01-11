# Reference: https://arxiv.org/pdf/1409.0473.pdf

from keras.models import Model
from keras.layers import Input, LSTM, Dot, Concatenate, Layer, Embedding, Bidirectional, Dense, Activation, RepeatVector, Lambda
from keras.activations import softmax
import tensorflow as tf
import numpy as np

# Attention


class Attention:
    def __init__(self, max_len_input, dense_size=10):
        self.max_len_input = max_len_input
        self.dense_1 = Dense(dense_size, activation='tanh',
                             name='AttentionDense_1')
        self.dense_2 = Dense(1, name='AttentionDense_2')
        self.concatenate = Concatenate(axis=-1, name='AttentionConcat')
        self.repeatvector = RepeatVector(max_len_input, name='AttentionRepeat')
        self.dot = Dot(axes=1, name='AttentionDot')
        self.softmax_over_time = Lambda(lambda x: softmax(
            x, axis=1), name='AttentionSoftMaxOverTime')

    def forward(self, st_1, h):
        st_1 = self.repeatvector(st_1)  # (L1, M2)
        x = self.concatenate([h, st_1])  # (L1, 2*M1+M2)
        x = self.dense_1(x)  # (L1, dense_size)
        x = self.dense_2(x)  # (L1, 1)
        x = self.softmax_over_time(x)  # (L1, 1)
        return self.dot([x, h])  # (1, 2*M1)

# Decoder


class Decoder:
    def __init__(self, max_len_input, max_words_target, hidden_dim):
        self.max_words_target = max_words_target
        self.dense = Dense(
            max_words_target, name='DecoderDense', activation='softmax')
        self.attention = Attention(max_len_input)
        self.hidden_dim = hidden_dim
        self.concatenate = Concatenate(axis=-1, name='DecoderConcat')
        self.lstm = LSTM(hidden_dim, return_state=True, name='DecoderRnn')

    def forward(self, encoder_outputs, decoder_inputs, s, c, target_len, inference=False):
        probs = []

        for t in range(target_len):
            context = self.attention.forward(s, encoder_outputs)  # (1, 2*M1)
            selector = Lambda(lambda x: x[:, t:t+1], name="Selector_" + str(t))
            xt = selector(decoder_inputs)  # (1, E2)
            xt = self.concatenate([context, xt])  # (1, E2+2*M1)

            o, s, c = self.lstm(xt, initial_state=[s, c])  # (M2)
            prob = self.dense(o)  # (W2)
            probs.append(prob)

        if(inference):
            # Get the last prob, hidden_state, cell_state
            return probs[-1], s, c
        else:
            return probs

# Stack and transpose


def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = tf.stack(x) # is now T x batch_size x output_vocab_size tensor
    x = tf.transpose(x, perm=[1, 0, 2]) # is now batch_size x T x output_vocab_size
    return x


StackLayer = Lambda(stack_and_transpose, name='Stack')

# A seprate inference model is required because decoder input's length is 1 instead of MAX_LEN_TARGET in training


class InferenceModel():
    def __init__(self, encoder_model, decoder_model):
        ''' A separate inference model is required because decoder input's length is 1 instead of MAX_LEN_TARGET in training
        '''
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def predict(self, input_sequence, sos_key, eos_key, max_len=100):
        ''' Predict the input sequence
        '''
        hidden_dim_decoder = self.decoder_model.input_shape[2][1]
        encoder_outputs = self.encoder_model.predict(input_sequence)
        s = np.zeros((1, hidden_dim_decoder))
        c = np.zeros((1, hidden_dim_decoder))
        target_seq = np.array([sos_key]).reshape(1, 1)
        max_len = 100
        sequence = []
        for _ in range(max_len):
            o, s, c = self.decoder_model.predict(
                [encoder_outputs, target_seq, s, c])
            # TODO: Add beam search here based on probability
            pred = np.argmax(o)
            sequence.append(pred)
            if(pred == eos_key):
                break
            target_seq[0, 0] = pred

        return sequence


class ModelFactory:
    # Create train and inference model
    @staticmethod
    def create(max_len_input,
               max_len_target,
               embedding_dim_encoder,
               embedding_dim_decoder,
               hidden_dim_encoder,
               hidden_dim_decoder,
               max_words_input,
               max_words_target):

        encoder_input = Input(shape=(max_len_input,), name='EncoderInput')
        decoder_input = Input(shape=(max_len_target,), name='DecoderInput')
        initial_s_input = Input(
            shape=(hidden_dim_decoder,), name='DecoderHiddenState')
        initial_c_input = Input(
            shape=(hidden_dim_decoder,), name='DecoderCellState')

        # Create train model

        # Encoder
        encoder_embedding_layer = Embedding(
            max_words_input, embedding_dim_encoder, input_length=max_len_input, name='EncoderEmbedding')
        encoder_embedding_output = encoder_embedding_layer(encoder_input)
        encoder_birnn_layer = Bidirectional(
            LSTM(hidden_dim_encoder, return_sequences=True), name='EncoderBiRnn')
        encoder_outputs = encoder_birnn_layer(encoder_embedding_output)

        # Decoder
        decoder_embedding_layer = Embedding(
            max_words_target, embedding_dim_decoder, name='DecoderEmbedding')
        decoder_embedding_output = decoder_embedding_layer(decoder_input)
        decoder = Decoder(max_len_input, max_words_target, hidden_dim_decoder)

        decoder_outputs = decoder.forward(
            encoder_outputs, decoder_embedding_output, initial_s_input, initial_c_input, max_len_target)

        # stack outputs
        probs = StackLayer(decoder_outputs)

        # Train model
        model = Model(inputs=[encoder_input, decoder_input,
                              initial_s_input, initial_c_input], outputs=probs)

        # Create inference model
        # Run once per generation
        encoder_model = Model(inputs=encoder_input, outputs=encoder_outputs)

        encoder_outputs_as_input = Input(
            shape=(max_len_input, 2*hidden_dim_encoder))
        decoder_single_input = Input(shape=(1,))
        decoder_single_embedding_output = decoder_embedding_layer(
            decoder_single_input)

        decoder_single_output, decoder_single_h, decoder_single_c = decoder.forward(
            encoder_outputs_as_input, decoder_single_embedding_output, initial_s_input, initial_c_input, 1, True)
        decoder_model = Model(inputs=[encoder_outputs_as_input, decoder_single_input, initial_s_input, initial_c_input], outputs=[
                              decoder_single_output, decoder_single_h, decoder_single_c])

        return model, InferenceModel(encoder_model, decoder_model)
