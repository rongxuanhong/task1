from keras.layers import Activation, Conv2D, Concatenate, AveragePooling2D, Dense, ZeroPadding2D, \
    GlobalAveragePooling2D, MaxPooling2D, Input, Add
from keras.models import Model
import keras.backend as backend
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from switch_norm import SwitchNormalization


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    # print(int(backend.int_shape(x)[1]))
    bn_axis = 1
    x = SwitchNormalization(axis=bn_axis,
                            name=name + '_sn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
               use_bias=False,
               data_format='channels_first',
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool', data_format='channels_first', )(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 1
    # x1 = SwitchNormalization(axis=bn_axis, name=name + '_0_bn')(x)
    # x1 = Activation('relu', name=name + '_0_relu')(x1)
    # x1 = Conv2D(4 * growth_rate, 1,
    #             use_bias=False,
    #             data_format='channels_first',
    #             name=name + '_1_conv')(x1)
    x1 = SwitchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                name=name + '_2_conv',
                kernel_initializer='he_uniform')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks, input_shape, classes_num, ):
    img_input = Input(shape=input_shape)

    bn_axis = 1

    x = Conv2D(64, 7, strides=1, use_bias=False, padding='same', data_format='channels_first', name='conv1/conv',
               kernel_initializer='he_uniform', )(
        img_input)
    x = SwitchNormalization(axis=bn_axis, name='conv1/sn')(x)

    x = Activation('relu', name='conv1/relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same', data_format='channels_first', name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.8, name='pool2')

    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.8, name='pool3')

    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.8, name='pool4')

    x = dense_block(x, blocks[3], name='conv5')

    x = SwitchNormalization(axis=bn_axis, name='sn')(x)
    x = Activation('relu', name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool', data_format='channels_first', )(x)
    x = Dense(classes_num, name='fc')(x)
    x = Activation('softmax', name='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    return model


if __name__ == '__main__':
    model = DenseNet([5, 5, 5, 5], (3, 320, 64), 10)
    model.summary()
