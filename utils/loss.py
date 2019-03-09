import keras.backend as K


def categorical_focal_loss(gamma=3.):
    def focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred` [batch_size, nb_class] one-hot
        :param y_pred: A tensor resulting from a softmax [batch_size, nb_class]
        :return: Output tensor.
        """
        y_pred /= K.sum(y_pred, len(y_pred.get_shape()) - 1, keepdims=True)
        epsilon = K.epsilon()
        # # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Cross entropy
        ce = y_true * K.log(y_pred)  ##  element_wise multipy

        weight = K.pow(1 - y_pred, gamma)  # 在多分类中alpha参数是没有效果的，因为类别平衡

        # Now fl has a shape of [batch_size, nb_class]
        fl = weight * ce

        # Both reduce_sum and reduce_max are ok
        reduce_fl = -K.sum(fl, axis=-1)
        return reduce_fl

    return focal_loss
