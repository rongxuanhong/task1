from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation, GlobalMaxPooling2D, Dropout, SeparableConv2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, )

from keras import backend as K
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from switch_norm import SwitchNormalization


def BaselineCnn(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'

    input_layer = Input(shape=(seq_len, mel_bins))
    x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = Conv2D(64, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)

    x = Conv2D(128, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)

    x = Conv2D(256, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)

    x = Conv2D(512, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)

    x = GlobalMaxPooling2D(data_format=data_format)(x)
    output_layer = Dense(classes_num, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def VggishConvBlock(input, filters, data_format, hasmaxpool=True):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    # x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same',
    #            data_format=data_format)(input)
    # x = SwitchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
               data_format=data_format, activation='linear')(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if hasmaxpool:
        x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    return x


def Bottleneck(input, filters, data_format, activation='relu'):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    x = Conv2D(filters=filters, kernel_size=(1, 1), data_format=data_format, use_bias=False)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation(activation)(x)

    return x


def Vggish(seq_len, mel_bins, classes_num):  # 0.687(单通道)

    data_format = 'channels_first'

    # input_layer = Input(shape=(seq_len, mel_bins))
    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x_1 = VggishConvBlock(input=x, filters=256, data_format=data_format)
    x = VggishConvBlock(input=x_1, filters=512, data_format=data_format)

    x = GlobalAveragePooling2D(data_format=data_format)(x)
    x = Dense(classes_num)(x)
    # print(x.shape)

    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def pooling_shape(input_shape):
    print(input_shape)
    if isinstance(input_shape, list):
        (sample_num, c, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, c, time_steps, freq_bins) = input_shape

    return (sample_num, c)


def attention_pooling(input):
    x1, x2 = input
    x2 = K.clip(x2, K.epsilon(), 1 - K.epsilon())
    p = x2 / K.sum(x2, axis=[2, 3])[..., None, None]
    return K.sum(x1 * p, axis=[2, 3])


def Vggish_multi_attention(seq_len, mel_bins, classes_num):  # 0.708 sigmoid linear不行

    data_format = 'channels_first'

    # input_layer = Input(shape=(seq_len, mel_bins))
    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer) densetnet+注意力？

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x1 = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x2 = VggishConvBlock(input=x1, filters=256, data_format=data_format, hasmaxpool=False)
    x3 = VggishConvBlock(input=x2, filters=512, data_format=data_format, hasmaxpool=False)  # 注意力2

    # x11 = Bottleneck(x1, classes_num, data_format=data_format, activation='sigmoid')
    # x12 = Bottleneck(x1, classes_num, data_format=data_format, activation='softmax')
    #
    # x1 = Lambda(attention_pooling, output_shape=pooling_shape, )([x11, x12])

    x21 = Bottleneck(x2, classes_num, data_format=data_format, activation='sigmoid')
    x22 = Bottleneck(x2, classes_num, data_format=data_format, activation='softmax')

    x2 = Lambda(attention_pooling, output_shape=pooling_shape, )([x21, x22])

    x31 = Bottleneck(x3, classes_num, data_format=data_format, activation='sigmoid')
    x32 = Bottleneck(x3, classes_num, data_format=data_format, activation='softmax')

    x3 = Lambda(attention_pooling, output_shape=pooling_shape, )([x31, x32])

    x = Concatenate(axis=-1)([x2, x3])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_single_attention(seq_len, mel_bins, classes_num):  # 0.729 softmax

    data_format = 'channels_first'

    # input_layer = Input(shape=(seq_len, mel_bins))
    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x1 = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x2 = VggishConvBlock(input=x1, filters=256, data_format=data_format)
    x3 = VggishConvBlock(input=x2, filters=512, data_format=data_format)  # 注意力2

    x11 = Bottleneck(x1, classes_num, data_format=data_format, activation='sigmoid')
    x12 = Bottleneck(x1, classes_num, data_format=data_format, activation='softmax')

    x1 = Lambda(attention_pooling, output_shape=pooling_shape, )([x11, x12])

    x21 = Bottleneck(x2, classes_num, data_format=data_format, activation='sigmoid')
    x22 = Bottleneck(x2, classes_num, data_format=data_format, activation='softmax')

    x2 = Lambda(attention_pooling, output_shape=pooling_shape, )([x21, x22])

    x31 = Bottleneck(x3, classes_num, data_format=data_format, activation='sigmoid')
    x32 = Bottleneck(x3, classes_num, data_format=data_format, activation='softmax')

    x3 = Lambda(attention_pooling, output_shape=pooling_shape, )([x31, x32])

    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_attention(seq_len, mel_bins, classes_num):  # 0.738 softmax

    data_format = 'channels_first'

    # input_layer = Input(shape=(seq_len, mel_bins))
    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x = VggishConvBlock(input=x, filters=256, data_format=data_format)
    x = VggishConvBlock(input=x, filters=512, data_format=data_format)

    # x = Bottleneck(x, 256, data_format=data_format)
    # x1 = Dense(classes_num, activation='sigmoid')(x)
    # x2 = Dense(classes_num, activation='softmax')(x)
    # x1 = Bottleneck(x, 128, data_format=data_format, activation='sigmoid')
    # x2 = Bottleneck(x, 128, data_format=data_format, activation='softmax')
    x1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x)
    x1 = Activation('sigmoid')(x1)
    x2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x)
    x2 = Activation('softmax')(x2)
    x2 = Lambda(lambda x: K.log(x), )(x2)

    x = Lambda(attention_pooling, output_shape=pooling_shape, )([x1, x2])

    # print(x.shape)
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_base_single_attention(seq_len, mel_bins, classes_num):  # 0.687(单通道)

    data_format = 'channels_first'
    # input_layer = Input(shape=(seq_len, mel_bins))
    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format, )
    x_1 = VggishConvBlock(input=x, filters=256, data_format=data_format, )
    x = VggishConvBlock(input=x_1, filters=512, data_format=data_format, )

    x1 = Conv2D(filters=classes_num, kernel_size=(1, 1), data_format=data_format, activation='sigmoid')(x)
    x2 = Conv2D(filters=classes_num, kernel_size=(1, 1), data_format=data_format, activation='softmax')(x)
    x = Lambda(attention_pooling, output_shape=pooling_shape, )([x1, x2])

    model = Model(inputs=input_layer, outputs=x)

    return model


if __name__ == '__main__':
    model = Vggish_attention(320, 64, 10)
    model.summary()
    # K.clear_session()
    # import tensorflow as tf

    # model = tf.keras.applications.resnet50.
    # model.summary()
