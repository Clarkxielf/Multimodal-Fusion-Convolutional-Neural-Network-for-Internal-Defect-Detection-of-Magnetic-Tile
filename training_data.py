import torch
import scipy.io as sio
import os
import glob
import numpy as np

label_dict = {0:'N', 1:'D'}

# 读取数据
def load_data(LABEL, split_ratio , Sampling_Interval ):
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

    all_Xtrain = []
    all_Labeltrain = []
    all_Xtest = []
    all_Labeltest = []

    for mat_name in glob.glob(os.path.join(DATA_DIR, '*%s.mat' % LABEL)):
        f = sio.loadmat(mat_name)
        data = f['data'][:,:7000].astype('float32')
        if LABEL == 'N':
            label = np.zeros(data.shape[0]).astype('int64')
        elif LABEL=='D':
            label = np.ones(data.shape[0]).astype('int64')

        sample_sequence = [Sampling_Interval * i for i in list(range(0, data.shape[1] // Sampling_Interval))]
        data = data.T[sample_sequence].T

        sample_train = [int((1 / split_ratio) * i) for i in list(range(0, int(split_ratio * data.shape[0])))]

        all_Xtrain.append(data[sample_train])
        all_Labeltrain.append(label[sample_train])
        all_Xtest.append(np.delete(data, sample_train, 0))
        all_Labeltest.append(np.delete(label, sample_train, 0))

    all_Xtrain = torch.from_numpy(np.concatenate(all_Xtrain, axis=0))
    all_Labeltrain = torch.from_numpy(np.concatenate(all_Labeltrain, axis=0))
    all_Xtest = torch.from_numpy(np.concatenate(all_Xtest, axis=0))
    all_Labeltest = torch.from_numpy(np.concatenate(all_Labeltest, axis=0))

    return all_Xtrain, all_Labeltrain, all_Xtest, all_Labeltest


