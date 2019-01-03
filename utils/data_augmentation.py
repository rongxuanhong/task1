import numpy as np


## random erasing
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_h=1 / 0.3, v_l=0, v_h=255, ):
    """

    :param p:擦除概率
    :param s_l:擦除矩形面积比下界
    :param s_h:擦除矩形面积比上界
    :param r_1:宽高比下界
    :param r_h:宽高比上界
    :param v_l: 替换的随机值下界
    :param v_h:  替换的随机值上界
    :return: eraser func
    reference paper:https://arxiv.org/pdf/1708.04896.pdf
    """

    def eraser(input_img):

        img_c, img_h, img_w = input_img.shape
        p_1 = np.random.rand()  # 随机概率

        if p_1 > p:  # 擦除概率大于0.5，不擦除
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w  # 确定随机的面积比例确定擦除的面积大小
            r = np.random.uniform(r_1, r_h)  # 确定随机的宽高比
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))  # 得到待擦除的矩形宽 w 高 h

            # 随机确定擦除矩阵的第一个坐标
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:  # 擦除的矩形保证在图片范围之内
                break

        c = np.random.uniform(v_l, v_h)

        input_img[:, top:top + h, left:left + w] = c  # 以c擦除

        return input_img

    return eraser
