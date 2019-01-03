import tensorflow as tf

"""
Tensorflow实现何凯明的Focal Loss, 该损失函数主要用于解决分类问题中的类别不平衡
focal_loss_sigmoid: 二分类loss
focal_loss_softmax: 多分类loss
Reference Paper : Focal Loss for Dense Object Detection
"""


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    L = -labels * (1 - alpha) * ((1 - y_pred) * gamma) * tf.log(y_pred) - \
        (1 - labels) * alpha * (y_pred ** gamma) * tf.log(1 - y_pred)
    return L


def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    L = tf.reduce_sum(L, axis=1)
    return L


def focal_loss(y_true, y_pred):
    # pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    # pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
    #     (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    # 返回每個样本的loss
    return tf.reduce_sum(-y_true * ((1 - y_pred) ** 2) * tf.log(y_pred), axis=1)


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, ):
    """

    :param p:
    :param s_l:
    :param s_h:
    :param r_1:
    :param r_2:
    :param v_l: 替换的随机值下界
    :param v_h:  替换的随机值上界
    :param pixel_level:
    :param data_format:
    :return:
    """

    def eraser(input_img):

        img_h, img_w = input_img.shape
        p_1 = np.random.rand()  # 随机概率

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w  # 确定随机的面积比例确定擦除的面积大小
            r = np.random.uniform(r_1, r_2)  # 确定随机的宽高比
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))  # 得到待擦除的矩形宽 w 高 h

            # 随机确定擦除矩阵的第一个坐标
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:  # 擦除的矩形保证在图片范围之内
                break

        c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, ] = c  # 以c擦除

        return input_img

    return eraser


if __name__ == '__main__':
    lr=1e-3
    for i in range(48):
        lr*=0.9
        print(lr)
