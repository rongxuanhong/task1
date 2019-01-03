from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, \
    Dense, Dropout, MaxPool2D, GlobalAveragePooling2D, Input, Activation, Concatenate, SeparableConv2D

from keras.regularizers import l2
from keras.models import Model
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from switch_norm import SwitchNormalization


class ConvBlock:
    """bn->relu->-conv"""

    def __init__(self, growth_rate, data_format, bottleneck, weight_decay=1e-4
                 , dropout_rate=0):
        self.bottleneck = bottleneck
        axis = -1 if data_format == 'channels_last' else 1
        inter_filter = growth_rate * 4
        self.conv2 = Conv2D(growth_rate,
                            (3, 3),
                            padding='same',
                            use_bias=False,
                            data_format=data_format,
                            kernel_initializer='he_uniform',
                            # kernel_regularizer=l2(weight_decay)
                            )
        # 初始化本模块所需要的op
        self.batchnorm1 = SwitchNormalization(axis=axis)
        # self.dropout = Dropout(dropout_rate)

        if self.bottleneck:
            self.conv1 = Conv2D(inter_filter,
                                (1, 1),
                                padding='same',
                                use_bias=False,
                                data_format=data_format,
                                kernel_initializer='he_uniform',
                                # kernel_regularizer=l2(weight_decay)
                                )
            self.batchnorm2 = SwitchNormalization(axis=axis)

    def build(self, x):

        output = self.batchnorm1(x, )

        if self.bottleneck:
            output = Activation('relu')(output)
            output = self.conv1(output)
            output = self.batchnorm2(output)

        output = Activation('relu')(output)
        output = self.conv2(output)
        # output = self.dropout(output)
        return output


class TransitionBlock:
    """transition block to reduce the number of filters"""

    def __init__(self, num_filters, data_format,
                 weight_decay=1e-4, dropout_rate=0.):
        axis = -1 if data_format == 'channels_last' else 1
        self.batchnorm = SwitchNormalization(axis=axis)

        self.conv = Conv2D(num_filters,
                           (1, 1),
                           padding='same',
                           use_bias=False,
                           data_format=data_format,
                           kernel_initializer='he_uniform',
                           # kernel_regularizer=l2(weight_decay),
                           )
        self.avg_pool = AveragePooling2D(data_format=data_format)

    def build(self, x):
        #### 这里没有加 dropout ###
        output = self.batchnorm(x)
        output = Activation('relu')(output)
        output = self.conv(output)
        output = self.avg_pool(output)
        return output


class DenseBlock:
    def __init__(self, num_layers, growth_rate, data_format, bottleneck,
                 weight_decay=1e-4, dropout_rate=0):
        self.num_layers = num_layers
        self.axis = -1 if data_format == 'channels_last' else 1
        self.blocks = []  # save each convblock in each denseblock
        for _ in range(int(self.num_layers)):
            self.blocks.append(ConvBlock(growth_rate, data_format, bottleneck, weight_decay,
                                         dropout_rate))

    def build(self, x):
        # concate each convblock within denseblock to get output of denseblock
        for i in range(int(self.num_layers)):
            output = self.blocks[i].build(x)
            x = Concatenate(axis=self.axis)([x, output])

        return x


class DenseNet:
    def __init__(self, depth_of_model, growth_rate, num_of_blocks,
                 output_classes, num_layers_in_each_block, data_format='channels_first',
                 bottleneck=True, compression=0.5, weight_decay=1e-4,
                 dropout_rate=0., pool_initial=True, include_top=True):
        self.depth_of_model = depth_of_model  # valid when num_layers_in_each_block is integer
        self.growth_rate = growth_rate
        self.num_of_blocks = num_of_blocks
        self.output_classes = output_classes
        self.num_layers_in_each_block = num_layers_in_each_block  # list tuple or integer
        self.data_format = data_format
        self.bottleneck = bottleneck
        self.compression = compression  # compression factor
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.pool_initial = pool_initial
        self.include_top = include_top

        # 决定每个block的层数
        if isinstance(self.num_layers_in_each_block, list) or isinstance(
                self.num_layers_in_each_block, tuple):  # 指定每个blocks的层数
            self.num_layers_in_each_block = list(self.num_layers_in_each_block)
        else:
            if self.num_layers_in_each_block == -1:  # 由模型深度决定每个block的层数
                if self.num_of_blocks != 3:
                    raise ValueError(
                        'Number of blocks must be 3 if num_layers_in_each_block is -1')
                if (self.depth_of_model - 4) % 3 == 0:
                    num_layers = (self.depth_of_model - 4) / 3
                    if self.bottleneck:
                        num_layers //= 2
                    self.num_layers_in_each_block = [num_layers] * self.num_of_blocks
                else:
                    raise ValueError("Depth must be 3N+4 if num_layer_in_each_block=-1")
            else:  # 每个blocks的层数相同
                self.num_layers_in_each_block = [self.num_layers_in_each_block] * self.num_of_blocks

        axis = -1 if data_format == 'channels_last' else 1

        # setting the filters and stride of the initial covn layer.
        # if self.pool_initial:
        # init_filters = (7, 7)
        # stride = (2, 2)
        # else:
        #     init_filters = (3, 3)
        #     stride = (1, 1)
        init_filters = (5, 5)
        stride = (1, 1)
        self.num_filters = 2 * self.growth_rate

        # 定义第一个conv以及pool
        self.conv1 = Conv2D(64,
                            init_filters,
                            strides=stride,
                            padding='same',
                            use_bias=False,
                            data_format=self.data_format,
                            kernel_initializer='he_uniform',
                            # kernel_regularizer=l2(self.weight_decay)
                            )

        if self.pool_initial:
            self.pool1 = MaxPool2D(pool_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   data_format=self.data_format)
            self.batchnorm1 = SwitchNormalization(axis=axis)
        self.batchnorm2 = SwitchNormalization(axis=axis)

        # last pool and fc layer
        # if self.include_top:  # is need top layer
        self.last_pool = GlobalAveragePooling2D(data_format=self.data_format)
        self.classifier = Dense(self.output_classes)

        # calculate the number of filters after each denseblock
        num_filters_after_each_block = [self.num_filters]
        for i in range(1, self.num_of_blocks):
            temp_num_filters = num_filters_after_each_block[i - 1] + \
                               self.growth_rate * self.num_layers_in_each_block[i - 1]
            num_filters_after_each_block.append(int(temp_num_filters * self.compression))  # compress filters

        # dense block initialization
        self.dense_block = []
        self.transition_blocks = []
        for i in range(self.num_of_blocks):
            self.dense_block.append(DenseBlock(self.num_layers_in_each_block[i],
                                               self.growth_rate,
                                               self.data_format,
                                               self.weight_decay,
                                               self.dropout_rate))
            if i + 1 < self.num_of_blocks:
                self.transition_blocks.append(TransitionBlock(num_filters_after_each_block[i],
                                                              self.data_format,
                                                              self.weight_decay,
                                                              self.dropout_rate))

    def build(self, input_shape):
        """ general modelling of DenseNet"""
        input = Input(shape=input_shape)
        output = self.conv1(input)
        if self.pool_initial:
            output = self.batchnorm1(output)
            output = Activation('relu')(output)
            output = self.pool1(output)

        for i in range(self.num_of_blocks - 1):
            output = self.dense_block[i].build(output)
            output = self.transition_blocks[i].build(output)

        output = self.dense_block[self.num_of_blocks - 1].build(output)  # output of the last denseblock
        output = self.batchnorm2(output)
        output = Activation('relu')(output)

        # if self.include_top:
        output = self.last_pool(output)
        output = self.classifier(output)

        output = Activation('softmax')(output)

        return Model(inputs=[input], outputs=[output], name='densenet')


def main():
    import tensorflow as tf
    model = DenseNet(190, 32, 4, 10, 5,
                     bottleneck=True, compression=0.5, weight_decay=1e-4, dropout_rate=0.2, pool_initial=True,
                     include_top=True)
    model = model.build(input_shape=(2, 320, 64))

    # model = tf.keras.applications.DenseNet121
    model.summary()


if __name__ == '__main__':
    main()
