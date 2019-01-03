from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation, GlobalMaxPooling2D, Dropout, SeparableConv2D, Add,
                          GlobalAveragePooling2D, Concatenate, GRU)

from keras import backend as K
from keras.regularizers import l2
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


# def VggishConvBlock(input, filters, data_format):
#     if data_format == 'channels_first':
#         bn_axis = 1
#
#     else:
#         raise Exception('Only support channels_first now!')
#
#     x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same', data_format=data_format)(input)
#     x = BatchNormalization(axis=bn_axis)(x)
#     x = Activation('relu')(x)
#     # x = Dropout(0.2)(x)
#
#     x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same', data_format=data_format)(x)
#     x = BatchNormalization(axis=bn_axis)(x)
#     x = Activation('relu')(x)
#     # x = Dropout(0.2)(x)
#     output = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
#
#     return output
def VggishConvBlock1(input, filters, data_format):
    if data_format == 'channels_first':
        bn_axis = 1

    else:
        raise Exception('Only support channels_first now!')

    x = SeparableConv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same', use_bias=False,
                        data_format=data_format)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    return x


def VggishConvBlock(input, filters, data_format, has_maxpool=True):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    # x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same',
    #            data_format=data_format)(input)
    # x = SwitchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)

    ## random erase 多卷积一层
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same',
               data_format=data_format, )(input)
    x = SwitchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    return x


def ConvBN(input, filters, data_format, has_maxPool=False, kernel_size=(3, 3), padding='same'):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding=padding, activation='linear',
               use_bias=False,
               data_format=data_format, )(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    if has_maxPool:
        x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    return x


def Vggish(seq_len, mel_bins, classes_num):  # 0.687(单通道)

    data_format = 'channels_first'

    input_layer = Input(shape=(3, seq_len, mel_bins))
    # x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x_1 = VggishConvBlock(input=x, filters=256, data_format=data_format)
    x = VggishConvBlock(input=x_1, filters=512, data_format=data_format)

    x = GlobalAveragePooling2D(data_format=data_format)(x)
    x = Dense(classes_num, )(x)

    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    import torch
    import torch.nn as nn

    return model


def Vggish3(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'

    input_layer = Input(shape=(seq_len, mel_bins))
    x = Reshape((1, seq_len, mel_bins))(input_layer)

    x = ConvBN(x, 42, data_format=data_format, kernel_size=(5, 5))
    x = ConvBN(x, 42, data_format=data_format, has_maxPool=True)
    x = ConvBN(x, 84, data_format=data_format)
    x = ConvBN(x, 84, data_format=data_format, has_maxPool=True)
    x = ConvBN(x, 168, data_format=data_format, has_maxPool=False)
    x = ConvBN(x, 168, data_format=data_format, has_maxPool=False)
    x = ConvBN(x, 168, data_format=data_format, has_maxPool=False)
    x = ConvBN(x, 168, data_format=data_format, has_maxPool=True)
    x = ConvBN(x, 336, data_format=data_format, has_maxPool=False, padding='valid')

    x = Conv2D(336, kernel_size=(1, 1), strides=(2, 2), use_bias=False, data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(10, kernel_size=(1, 1), strides=(2, 2), use_bias=False, data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)

    x = GlobalAveragePooling2D(data_format=data_format)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=input_layer, outputs=x)

    return model


def Xception(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    # weight_decay = 1e-4 # 0.653
    weight_decay = 1e-3  # 0.653

    bn_axis = 1
    input_layer = Input(shape=(seq_len, mel_bins))
    x = Reshape((1, seq_len, mel_bins))(input_layer)
    x = Conv2D(32, (3, 3),
               strides=(2, 2),
               use_bias=False,
               data_format=data_format,
               kernel_regularizer=l2(weight_decay),
               name='block1_conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2', data_format=data_format,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block1_conv2_bn', axis=bn_axis)(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1),
                      strides=(2, 2),
                      padding='same',
                      data_format=data_format,
                      use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    x = SeparableConv2D(128, (3, 3),
                        padding='same',
                        use_bias=False,
                        data_format=data_format,
                        kernel_regularizer=l2(weight_decay),
                        name='block2_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)

    x = Dropout(0.2)(x)
    x = SeparableConv2D(128, (3, 3),
                        padding='same',
                        data_format=data_format,
                        kernel_regularizer=l2(weight_decay),
                        use_bias=False,
                        name='block2_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                     strides=(2, 2),
                     padding='same',
                     data_format=data_format,
                     name='block2_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      data_format=data_format,

                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)

    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3),
                        padding='same',
                        use_bias=False,
                        data_format=data_format,
                        kernel_regularizer=l2(weight_decay),
                        name='block3_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)

    x = Dropout(0.2)(x)
    x = SeparableConv2D(256, (3, 3),
                        data_format=data_format,
                        padding='same',
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='block3_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same',
                     data_format=data_format,
                     name='block3_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(512, (1, 1),
                      strides=(2, 2),
                      data_format=data_format,
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)

    x = Dropout(0.2)(x)

    x = SeparableConv2D(512, (3, 3),
                        padding='same',
                        data_format=data_format,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='block4_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)

    x = Dropout(0.2)(x)

    x = SeparableConv2D(512, (3, 3),
                        padding='same',
                        data_format=data_format,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='block4_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same',
                     data_format=data_format,
                     name='block4_pool')(x)
    x = Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)

        x = Dropout(0.2)(x)
        x = SeparableConv2D(512, (3, 3),
                            padding='same',
                            data_format=data_format,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(512, (3, 3),
                            padding='same',
                            use_bias=False,
                            data_format=data_format,
                            kernel_regularizer=l2(weight_decay),
                            name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(512, (3, 3),
                            padding='same',
                            data_format=data_format,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv3_bn')(x)

        x = Add()([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      data_format=data_format,
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=bn_axis)(residual)

    x1 = GlobalAveragePooling2D(name='avg_pool1', data_format=data_format, )(x)

    x = Activation('relu', name='block13_sepconv1_act')(x)

    x = Dropout(0.2)(x)
    x = SeparableConv2D(512, (3, 3),
                        padding='same',
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        data_format=data_format,
                        name='block13_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)

    x = Dropout(0.2)(x)
    x = SeparableConv2D(1024, (3, 3),
                        padding='same',
                        data_format=data_format,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='block13_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                     strides=(2, 2),
                     data_format=data_format,
                     padding='same',
                     name='block13_pool')(x)
    x = Add()([x, residual])

    x2 = GlobalAveragePooling2D(name='avg_pool2', data_format=data_format, )(x)

    x = SeparableConv2D(1536, (3, 3),
                        padding='same',
                        data_format=data_format,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='block14_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = Dropout(0.2)(x)
    x = SeparableConv2D(2048, (3, 3),
                        padding='same',
                        data_format=data_format,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='block14_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D(name='avg_pool', data_format=data_format, )(x)
    x = Concatenate()([x1, x2, x])
    x = Dense(512, kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    output = Dense(classes_num, activation='softmax', name='predictions')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model


if __name__ == '__main__':
    # model, x = Vggish(320, 64, 10)
    # K.clear_session()
    import tensorflow as tf

    model = tf.keras.applications.resnet50.ResNet50()
    model.summary()
