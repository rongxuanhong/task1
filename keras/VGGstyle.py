from keras.layers import BatchNormalization, Conv2D, \
    Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, GaussianNoise, Activation, Input, Reshape
from keras.regularizers import l2
from keras.models import Model


def ConvBlock1(input, filters, initializer='he_uniform', weight_decay=5e-4, data_format='channels_first'):
    output = Conv2D(kernel_size=3,
                    filters=filters,
                    padding='same',
                    strides=1,
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay))(input)
    output = Activation('relu')(BatchNormalization(axis=1)(output))
    output = Conv2D(kernel_size=3,
                    filters=filters,
                    strides=1,
                    padding='same',
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay))(output)
    output = Activation('relu')(BatchNormalization(axis=1)(output))

    output = MaxPooling2D(pool_size=2,
                          data_format=data_format)(output)
    output = GaussianNoise(1.00)(output)
    return output


def ConvBlock2(input, filters, initializer='he_uniform', weight_decay=5e-4, data_format='channels_first'):
    output = Conv2D(kernel_size=3,
                    filters=filters,
                    padding='same',
                    strides=1,
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay))(input)
    output = Activation('relu')(BatchNormalization(axis=1)(output))
    output = Dropout(0.3)(output)  # 改为0.3

    output = Conv2D(kernel_size=3,
                    filters=filters,
                    strides=1,
                    padding='same',
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay))(output)
    output = Activation('relu')(BatchNormalization(axis=1)(output))
    output = Dropout(0.3)(output)

    output = Conv2D(kernel_size=3,
                    filters=filters,
                    strides=1,
                    padding='same',
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay))(output)
    output = Activation('relu')(BatchNormalization(axis=1)(output))
    output = Dropout(0.3)(output)

    output = Conv2D(kernel_size=3,
                    filters=filters,
                    strides=1,
                    padding='same',
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_decay))(output)
    output = Activation('relu')(BatchNormalization(axis=1)(output))
    output = Dropout(0.3)(output)

    output = MaxPooling2D(pool_size=2,
                          data_format=data_format)(output)
    output = GaussianNoise(0.75)(output)
    return output


def Vggish(seq_len, mel_bins, classes_num, initializer='he_uniform', weight_decay=1e-4):
    data_format = 'channels_first'

    input_layer = Input(shape=(seq_len, mel_bins))
    x = Reshape((1, seq_len, mel_bins))(input_layer)
    x = ConvBlock1(x, 48, initializer=initializer, weight_decay=weight_decay, data_format=data_format)
    x = ConvBlock1(x, 96, initializer=initializer, weight_decay=weight_decay, data_format=data_format)
    x = ConvBlock2(x, 192, initializer=initializer, weight_decay=weight_decay, data_format=data_format)

    x = Conv2D(kernel_size=3,
               filters=384,
               strides=1,
               data_format=data_format,
               use_bias=False,
               kernel_initializer=initializer,
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(BatchNormalization(axis=1)(x))
    x = Dropout(0.5)(x)

    x = Conv2D(kernel_size=1,
               filters=384,
               data_format=data_format,
               strides=1,
               use_bias=False,
               kernel_initializer=initializer,
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(BatchNormalization(axis=1)(x))
    x = Dropout(0.5)(x)

    x = Conv2D(kernel_size=1,
               filters=classes_num,
               strides=1,
               data_format=data_format,
               use_bias=False,
               kernel_initializer=initializer,
               kernel_regularizer=l2(weight_decay))(x)

    x = GlobalAveragePooling2D(data_format=data_format)(GaussianNoise(0.3)(x))
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def main():
    model = Vggish(320, 64, 10)
    model.summary()


if __name__ == '__main__':
    main()
