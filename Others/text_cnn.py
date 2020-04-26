from keras import Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Input
from model_define import AttentionLayer

def create_textcnn(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_matrix, trainable):
    kernel_sizes=[3, 4, 5]
    convs = []
    embedding_layers = Embedding(len(embeddings_matrix), EMBEDDING_DIM, weights=[
        embeddings_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=trainable)
    # 输入层
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # Embedding层
    x = embedding_layers(inputs)
    for kernel_size in kernel_sizes:
        c = Conv1D(128, kernel_size, activation='relu')(x)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    c = Concatenate()(convs)
    outputs = Dense(2, activation='sigmoid')(c)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量
    return model
