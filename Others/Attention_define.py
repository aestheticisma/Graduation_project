import numpy
import keras
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
K.clear_session()


class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def create_classify_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_matrix, HIDDEN_SIZE, ATTENTION_SIZE, trainable):
    embedding_layers = Embedding(len(embeddings_matrix), EMBEDDING_DIM, weights=[
        embeddings_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=trainable)
    # 输入层
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # Embedding层
    x = embedding_layers(inputs)
    # BiLSTM层
    x = Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, return_sequences=True))(x)
    # Attention层
    x = AttentionLayer(attention_size=ATTENTION_SIZE)(x)
    # 输出层
    x = BatchNormalization()(x)
    outputs = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量
    return model
