import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import config as cfg
import os
import torch


class MyDataset(Dataset):
    def __init__(self, train_or_test):
        # np加载数据
        self.data = np.loadtxt(os.path.join(cfg['data_path'], 'data.csv'))
        # 归一化
        self.data = self.data / cfg['normalize']
        # 数据放到内存
        self.data = torch.tensor(self.data, dtype=torch.float32)
        # 数据放到设备
        self.data = self.data.to(cfg['device'])
        if train_or_test == "train":
            self.data_index = np.load(os.path.join(cfg['data_path'], str(cfg['time_step']), 'train.npy'))
        else:
            self.data_index = np.load(os.path.join(cfg['data_path'], str(cfg['time_step']), 'test.npy'))

    # 会根据index返回数据
    def __getitem__(self, index):
        # index转化
        index = self.data_index[index]
        # 取数据
        x1 = self.data[index:index + 12]
        x2 = self.data[index + 11:index + 14]
        lable = self.data[index + 12:index + 15] * cfg['normalize']
        return x1, x2, lable

    def __len__(self):
        return self.data_index.shape[0]


# 获得mask矩阵
def get_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# 测试
if __name__ == "__main__":

    print(get_mask(4))

    exit()
    mydataset = MyDataset("test")

    train_loader2 = DataLoader(dataset=mydataset,
                               batch_size=1,
                               shuffle=True)

    for index, batch_data in enumerate(train_loader2):
        x1, x2, label = batch_data
        print(x1.shape)
        print(x2.shape)
        print(label.shape)
        exit()
