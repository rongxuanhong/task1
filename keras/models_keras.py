from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation, GlobalMaxPooling2D, UpSampling2D, SeparableConv2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, Conv2DTranspose)

from keras import backend as K
import sys
import os
from keras.regularizers import l2

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import keras


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


def VggishConvBlock(input, filters, data_format, hasmaxpool=True, weight_decay=5e-4):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    # x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same',
    #            data_format=data_format)(input)
    # x = SwitchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
               data_format=data_format, use_bias=False, kernel_regularizer=l2(5e-4))(
        input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    if hasmaxpool:
        x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    return x


def Res_block(input, filters):
    data_format = 'channels_first'
    bn_axis = 1
    res = input
    res = Conv2D(filters, 1, strides=2, padding='same', data_format=data_format, use_bias=False)(res)

    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False,
               data_format=data_format)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
               data_format=data_format, activation='linear')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    x = Add()([res, x])
    return x


def res_attention(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    bn_axis = 1

    input_layer = Input(shape=(3, seq_len, mel_bins))
    x = Res_block(input_layer, 64)
    x2 = Res_block(x, 128)
    x3 = Res_block(x2, 256)
    x4 = Res_block(x3, 512)

    upconv_x4 = UpSampling2D(data_format=data_format, )(x4)
    upconv_x4 = Conv2D(filters=256, kernel_size=(1, 1), data_format=data_format)(upconv_x4)
    upconv_x4 = BatchNormalization(axis=bn_axis)(upconv_x4)
    upconv_x4 = Activation('relu')(upconv_x4)
    x_34 = Add()([upconv_x4, x3])

    upconv_x3 = UpSampling2D(data_format=data_format, )(x3)
    upconv_x3 = Conv2D(filters=128, kernel_size=(1, 1), data_format=data_format)(upconv_x3)
    upconv_x3 = BatchNormalization(axis=bn_axis)(upconv_x3)
    upconv_x3 = Activation('relu')(upconv_x3)
    x_23 = Add()([upconv_x3, x2])

    x_1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_1 = BatchNormalization(axis=bn_axis)(x_1)
    x_1 = Activation('sigmoid')(x_1)
    x_2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_2 = BatchNormalization(axis=bn_axis)(x_2)
    x_2 = Activation('softmax')(x_2)
    x_2 = Lambda(lambda x: K.log(x), )(x_2)

    x_3 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_23 = BatchNormalization(axis=bn_axis)(x_23)
    x_3 = Activation('sigmoid')(x_3)
    x_4 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_4 = BatchNormalization(axis=bn_axis)(x_4)
    x_4 = Activation('softmax')(x_4)
    x_4 = Lambda(lambda x: K.log(x), )(x_4)

    x11 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_1, x_2])
    x22 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_3, x_4])

    x = Concatenate(axis=-1)([x11, x22])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


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


def Vggish_single_attention(seq_len, mel_bins, classes_num):  # 75.3+(1.5 SVM)=76.8?? 再跑试试 74.8+1.3=76.1
    data_format = 'channels_first'

    input_layer = Input(shape=(3, seq_len, mel_bins))
    weight_decay = 5e-4
    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x, filters=256, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x, filters=512, data_format=data_format, weight_decay=weight_decay)

    x1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('sigmoid')(x1)
    x2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x)
    x2 = BatchNormalization(axis=1)(x2)
    x2 = Activation('softmax')(x2)
    x2 = Lambda(lambda x: K.log(x), )(x2)

    x = Lambda(attention_pooling, output_shape=pooling_shape, )([x1, x2])

    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_attention_no_fcn(seq_len, mel_bins, classes_num):  # 0.753+(0.5 svm)=0.758

    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1

    else:
        raise Exception('Only support channels_first now!')

    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)
    ## 瓶颈层使用SN loss为nan

    weight_decay = 5e-4

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=weight_decay)
    x_2 = VggishConvBlock(input=x, filters=128, data_format=data_format, weight_decay=weight_decay)
    x_1 = VggishConvBlock(input=x_2, filters=256, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x_1, filters=512, data_format=data_format, weight_decay=weight_decay)

    x1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x)
    x1 = BatchNormalization(axis=bn_axis)(x1)
    x1 = Activation('sigmoid')(x1)
    x2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x)
    x2 = BatchNormalization(axis=bn_axis)(x2)
    x2 = Activation('softmax')(x2)
    x2 = Lambda(lambda x: K.log(x), )(x2)

    x3 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_1)
    x3 = BatchNormalization(axis=bn_axis)(x3)
    x3 = Activation('sigmoid')(x3)
    x4 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_1)
    x4 = BatchNormalization(axis=bn_axis)(x4)
    x4 = Activation('softmax')(x4)
    x4 = Lambda(lambda x: K.log(x), )(x4)

    x11 = Lambda(attention_pooling, output_shape=pooling_shape, )([x1, x2])
    x22 = Lambda(attention_pooling, output_shape=pooling_shape, )([x3, x4])

    x = Concatenate(axis=-1)([x11, x22])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_two_attention(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    input_layer = Input(shape=(3, seq_len, mel_bins))

    weight_decay = 5e-4

    x1 = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=weight_decay)
    x2 = VggishConvBlock(input=x1, filters=128, data_format=data_format, weight_decay=weight_decay)
    x3 = VggishConvBlock(input=x2, filters=256, data_format=data_format, weight_decay=weight_decay)
    x4 = VggishConvBlock(input=x3, filters=512, data_format=data_format, weight_decay=weight_decay)

    deconv_x4 = Conv2DTranspose(256, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x4)
    deconv_x4 = BatchNormalization(axis=bn_axis)(deconv_x4)
    deconv_x4 = Activation('relu')(deconv_x4)
    x_34 = Add()([deconv_x4, x3])

    deconv_x3 = Conv2DTranspose(128, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x3)  # 这里换成FPN形式:10000就基本拟合 73.8 非金字塔效果更好
    deconv_x3 = BatchNormalization(axis=bn_axis)(deconv_x3)
    deconv_x3 = Activation('relu')(deconv_x3)
    x_23 = Add()([deconv_x3, x2])

    x_1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_1 = BatchNormalization(axis=bn_axis)(x_1)
    x_1 = Activation('sigmoid')(x_1)

    x_2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_2 = BatchNormalization(axis=bn_axis)(x_2)
    x_2 = Activation('softmax')(x_2)
    x_2 = Lambda(lambda x: K.log(x), )(x_2)

    x_3 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_3 = BatchNormalization(axis=bn_axis)(x_3)
    x_3 = Activation('sigmoid')(x_3)

    x_4 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_4 = BatchNormalization(axis=bn_axis)(x_4)
    x_4 = Activation('softmax')(x_4)
    x_4 = Lambda(lambda x: K.log(x), )(x_4)

    x11 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_1, x_2])
    x22 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_3, x_4])

    x = Concatenate(axis=-1)([x11, x22])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_two_attention_up(seq_len, mel_bins, classes_num):  # 75.8(+-0.1)
    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    input_layer = Input(shape=(3, seq_len, mel_bins))

    weight_decay = 5e-4
    x1 = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=weight_decay)
    x2 = VggishConvBlock(input=x1, filters=128, data_format=data_format, weight_decay=weight_decay)
    x3 = VggishConvBlock(input=x2, filters=256, data_format=data_format, weight_decay=weight_decay)
    x4 = VggishConvBlock(input=x3, filters=512, data_format=data_format, weight_decay=weight_decay)

    upconv_x4 = UpSampling2D(data_format=data_format, )(x4)
    upconv_x4 = Conv2D(filters=256, kernel_size=(1, 1), data_format=data_format)(upconv_x4)
    upconv_x4 = BatchNormalization(axis=bn_axis)(upconv_x4)
    upconv_x4 = Activation('relu')(upconv_x4)
    x_34 = Add()([upconv_x4, x3])

    upconv_x3 = UpSampling2D(data_format=data_format, )(x3)
    upconv_x3 = Conv2D(filters=128, kernel_size=(1, 1), data_format=data_format)(upconv_x3)
    upconv_x3 = BatchNormalization(axis=bn_axis)(upconv_x3)
    upconv_x3 = Activation('relu')(upconv_x3)
    x_23 = Add()([upconv_x3, x2])

    x_1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_1 = BatchNormalization(axis=bn_axis)(x_1)
    x_1 = Activation('sigmoid')(x_1)
    x_2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_2 = BatchNormalization(axis=bn_axis)(x_2)
    x_2 = Activation('softmax')(x_2)
    x_2 = Lambda(lambda x: K.log(x), )(x_2)

    x_3 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_23 = BatchNormalization(axis=bn_axis)(x_23)
    x_3 = Activation('sigmoid')(x_3)
    x_4 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_4 = BatchNormalization(axis=bn_axis)(x_4)
    x_4 = Activation('softmax')(x_4)
    x_4 = Lambda(lambda x: K.log(x), )(x_4)

    x11 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_1, x_2])
    x22 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_3, x_4])

    x = Concatenate(axis=-1)([x11, x22])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_two_attentionFPN(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)

    x1 = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=5e-4)
    x2 = VggishConvBlock(input=x1, filters=128, data_format=data_format, weight_decay=5e-4)
    x3 = VggishConvBlock(input=x2, filters=256, data_format=data_format, weight_decay=5e-4)
    x4 = VggishConvBlock(input=x3, filters=512, data_format=data_format, weight_decay=5e-4)

    upconv_x4 = UpSampling2D(data_format=data_format, )(x4)
    upconv_x4 = Conv2D(filters=256, kernel_size=(1, 1), data_format=data_format)(upconv_x4)
    upconv_x4 = BatchNormalization(axis=bn_axis)(upconv_x4)
    upconv_x4 = Activation('relu')(upconv_x4)
    x_34 = Add()([upconv_x4, x3])

    upconv_x3 = UpSampling2D(data_format=data_format, )(x_34)
    upconv_x3 = Conv2D(filters=128, kernel_size=(1, 1), data_format=data_format)(upconv_x3)
    upconv_x3 = BatchNormalization(axis=bn_axis)(upconv_x3)
    upconv_x3 = Activation('relu')(upconv_x3)
    x_23 = Add()([upconv_x3, x2])

    x_1 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_1 = BatchNormalization(axis=bn_axis)(x_1)
    x_1 = Activation('sigmoid')(x_1)
    x_2 = Conv2D(filters=64, kernel_size=(1, 1), data_format=data_format)(x_34)
    x_2 = BatchNormalization(axis=bn_axis)(x_2)
    x_2 = Activation('softmax')(x_2)
    x_2 = Lambda(lambda x: K.log(x), )(x_2)

    x_3 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_23 = BatchNormalization(axis=bn_axis)(x_23)
    x_3 = Activation('sigmoid')(x_3)
    x_4 = Conv2D(filters=32, kernel_size=(1, 1), data_format=data_format)(x_23)
    x_4 = BatchNormalization(axis=bn_axis)(x_4)
    x_4 = Activation('softmax')(x_4)
    x_4 = Lambda(lambda x: K.log(x), )(x_4)

    x11 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_1, x_2])
    x22 = Lambda(attention_pooling, output_shape=pooling_shape, )([x_3, x_4])

    x = Concatenate(axis=-1)([x11, x22])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


if __name__ == '__main__':
    model = Vggish_two_attention_up(320, 64, 10)
    model.summary()
    # K.clear_session()
    # import tensorflow as tf

    # model = tf.keras.applications.resnet50.
    # model.summary()
