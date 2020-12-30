# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: model.py
# @Software: PyCharm


from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout
from keras.layers import Flatten, Dense, Activation, BatchNormalization, CuDNNGRU, CuDNNLSTM
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers.wrappers import Bidirectional



def base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Flatten()(merge)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def BiGRU_base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)

    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
