from torch import nn
from config import config as cfg
import torch

import copy
import transformer_moudles
import torch.nn as nn
from config import config as cfg
from load_data import get_mask


# transformer
class My_transformer(nn.Module):

    def __init__(self):
        super(My_transformer, self).__init__()
        # 复制器
        self.c = copy.deepcopy
        # 位置编码
        self.position = transformer_moudles.PositionalEncoding(d_model=cfg['d_model'], dropout=0,
                                                               max_len=cfg['d_model'])
        # 两个前向传播
        self.ff = transformer_moudles.PositionwiseFeedForward(cfg['d_model'], cfg['d_ff'], cfg['dropout'])
        # 多头注意力
        self.attn = transformer_moudles.MultiHeadedAttention(cfg['head'], cfg['d_model'])

        # 编码器
        self.encoder = transformer_moudles.Encoder(
            transformer_moudles.EncoderLayer(
                cfg['d_model'],
                self.c(self.attn),
                self.c(self.ff),
                cfg['dropout']),
            cfg['encoder_N'])

        # 解码器
        self.decoder = transformer_moudles.Decoder(
            transformer_moudles.DecoderLayer(
                cfg['d_model'],
                self.c(self.attn),
                self.c(self.attn),
                self.c(self.ff),
                cfg['dropout']),
            cfg['decoder_N'])

        # 最后一层
        self.last_layer = nn.Linear(cfg['d_model'], cfg['d_model']).to(cfg['device'])

    # transformer前向传播
    def forward(self, src, tgt, src_mask, tgt_mask):
        # 编码
        src = self.encode(self.position(src), src_mask)
        return self.decode(src, src_mask, self.position(tgt), tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return torch.relu(self.last_layer(self.decoder(tgt, memory, src_mask, tgt_mask)))


# LSTM
class LSTM(nn.Module):
    "Implements FFN equation."

    def __init__(self):
        super(LSTM, self).__init__()
        self.LSTM = nn.LSTM(input_size=cfg['d_model'],
                            hidden_size=cfg['hidden_size'],
                            num_layers=cfg['num_layers'],
                            bias=True,
                            batch_first=True,
                            dropout=cfg['LSTM_dropout'],
                            bidirectional=False).to(cfg['device'])
        # 输出全连接层
        self.last_layer = nn.Linear(cfg['hidden_size'], cfg['d_model']).to(cfg['device'])

    def forward(self, x1, x2, no1=None, no2=None):
        # x1和x2拼接
        x = torch.cat((x1, x2), dim=1).to(cfg['device'])

        # 初始化的h0和c0
        output, (_, _) = self.LSTM(x)  # 返回的（batch_size，time_step，hidensize）

        # 取最后三个
        output = output[:, 12:12 + cfg['time_step'], :]

        return torch.relu(self.last_layer(output))


#-------------------------------------本地测试--------------------------------------------

# LSTM测试
def LSTM_test():
    # 输入
    x1 = torch.ones(cfg['train_batch_size'], 12, cfg['d_model']).to(cfg['device'])  # 1-12
    x2 = torch.ones(cfg['train_batch_size'], 3, cfg['d_model']).to(cfg['device'])  # 12-14
    model = LSTM()
    output = model(x1, x2)
    print(output.shape)

# transformer测试
def transformer_test():
    # 输入
    # x1 = torch.ones(2, 12, cfg['d_model']).to(cfg['device'])  # 1-12
    # x2 = torch.ones(2, 3, cfg['d_model']).to(cfg['device'])  # 12-14
    #
    # # 输出
    # label = torch.ones(1, 3, cfg['d_model']).to(cfg['device'])  # 13-15
    #
    # # 定义模型
    # model = My_transformer().to(cfg['device'])
    #
    # # 初始化
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
    #
    # # 掩码
    # tgt_mask=get_mask(3).to(cfg['device'])
    #
    # # 预测
    # pre = model(x1, x2, None, tgt_mask)
    #
    # print(pre)

    # ------------------预测时候---------------------------
    x1 = torch.ones(1, 12, cfg['d_model']).to(cfg['device'])  # 1-12
    x2 = torch.ones(1, 3, cfg['d_model']).to(cfg['device'])  # 12-14

    # 输出
    label = torch.ones(1, 3, cfg['d_model']).to(cfg['device'])  # 13-15

    # 定义模型
    model = My_transformer().to(cfg['device'])

    # 初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 预测
    model.eval()
    feed = x2[:, 0:1, :]  # 第一个
    for time in range(3):
        # 预测
        pre = model(x1, feed, None, None)
        # 拿到第13+time的值
        if time == 0:
            pres = pre[:, time:time + 1, :]
        else:
            pres = torch.cat((pres, pre[:, time:time + 1, :]))
        # 并且把这一天的值作为新的输入
        feed = torch.cat((feed,  pre[:, time:time + 1, :] / cfg['normalize']), dim=1)

        print(feed.shape)

    print(pre.shape)

if __name__ == "__main__":
    transformer_test()





