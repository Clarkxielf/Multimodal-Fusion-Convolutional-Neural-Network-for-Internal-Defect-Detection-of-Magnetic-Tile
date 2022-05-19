import numpy as np
import torch
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from io import BytesIO
import PIL

# 对原始的声音数据做归一化处理
def Nor(x):
    x=x.numpy()
    DATA=[]
    for i in range(x.shape[0]):
        data=(x[i]-min(x[i])) / (max(x[i])-min(x[i]))
        DATA.append(data)
    X=np.stack(DATA,axis=0)
    X=torch.from_numpy(X)

    return X

# 对data_loader中每一个batch做傅里叶变换
def FFT(x):
    DATA = []
    x = x.numpy()
    x = abs(fft(x))[:,0:3500]
    for i in range(x.shape[0]):
        data = x[i] / max(x[i])
        DATA.append(data)
    y = np.stack(DATA, axis=0)
    y = torch.from_numpy(y)

    return y


# 对data_loader中每一个batch做语谱图
def Spectrogram(x):
    DATA=[]
    for i in range(x.size(0)):
        wavdata = x.numpy()[i, :]
        framerate=40000
        framelength = 0.025
        framesize = framelength * framerate
        nfftdict = {}
        lists = [32, 64, 128, 256, 512, 1024]

        for j in lists:
            nfftdict[j] = abs(framesize - j)
        sortlist = sorted(nfftdict.items(), key=lambda x: x[1])
        framesize = int(sortlist[0][0])
        NFFT = framesize
        overlapSize = 1.0 / 3 * framesize
        overlapSize = int(round(overlapSize))
        plt.figure(figsize=(4, 4), dpi=56)
        spectrum, freqs, ts, fig = plt.specgram(wavdata, NFFT=NFFT, Fs=framerate, window=np.hanning(M=framesize),
                                                noverlap=overlapSize, mode='default', scale_by_freq=True,
                                                sides='default',
                                                scale='dB', xextent=None)  # 绘制频谱图
        plt.axis('off')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())#去白边
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        buffer = BytesIO() #分配缓存
        plt.savefig(buffer, format='jpg')#将图片存入缓存
        plt.close()
        dataPIL = PIL.Image.open(buffer)
        data = np.asarray(dataPIL).transpose(2, 0, 1).astype(np.float32)/ 255
        buffer.close()#清除缓存
        DATA.append(data)


    z=np.stack(DATA,axis=0)
    z=torch.from_numpy(z)
    return z

