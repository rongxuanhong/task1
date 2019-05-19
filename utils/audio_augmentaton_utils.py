import librosa
import numpy as np


# y, sr = librosa.load('../airport-barcelona-0-0-a.wav', sr=48000, duration=10.0)


# 1.
def time_stretch(y, mode='up', rate=1.0):
    """
    对时间轴加速或减速
    :param mode:
    :param rate:
    :return:
    """
    if mode == 'up':
        y_stretch_up = librosa.effects.time_stretch(y, rate)
        return y_stretch_up
    else:
        y_stretch_down = librosa.effects.time_stretch(y.astype('float16'), rate)
        return y_stretch_down


# 2.
def add_gaussian_noise(y):
    """
    添加分布噪声,如高斯噪声
    :param y:
    :return:
    """
    noise_amap = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amap.astype(np.float32) * np.random.normal(size=len(y))
    return y_noise
    # 白噪
    # length = audio.shape[0]
    # print(length)
    # noise = np.random.randn(length)
    # noise = noise.astype(np.float32)  # 这里要进行类型转换，不然保存的时候会出现格式错误
    # noise -= np.mean(noise)
    # print(noise.shape)
    #
    # signal_power = np.dot(audio.T, audio) / length
    # noise_variance = signal_power / (10 ** (20 / 10))
    # noise = np.sqrt(noise_variance) / np.std(noise) * noise
    # y = audio + noise
    # return y


# print(y_noise)
# 3. 随机移动
def random_shifting(y):
    """
    随机移动 onset
    :param y:
    :return:
    """
    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
    start = int(y.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y, (start, 0), mode='constant')[0:y.shape[0]]  # 信号右移(数组前面填充start个0，再取前y.shape[0]个采样点)
    else:
        y_shift = np.pad(y, (0, -start), mode='constant')[0:y.shape[0]]  # 信号左移
    return y_shift


# 4. 基音变化
def pitch_shift(y, sr, n_steps=3.0, bins_per_octave=12):
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps, bins_per_octave=bins_per_octave)


def save_audio(save_path, audio, sr):
    """
    保存录音
    :param save_path: 保存路径
    :param audio: 声音的时间序列
    :param sr: 采样率
    :return:
    """
    # maxv = np.iinfo(np.float16).max
    return librosa.output.write_wav(save_path, audio, sr)
