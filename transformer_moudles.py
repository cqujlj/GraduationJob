import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from config import config as cfg


# 生成器：全连接 + softmax
class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# 克隆一个模块多次
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 编码器,堆叠的多个编码器单层
class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 把基础层复制了n层,是一个nn.ModuleList
        self.layers = clones(layer, N)
        # 层正则化
        self.norm = LayerNorm(layer.size)

    # 依次编码每一个，输出最后一层
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 层正则化
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features).to(cfg['device']))
        self.b_2 = nn.Parameter(torch.zeros(features).to(cfg['device']))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差连接：LayerNorm(x+Sublayer(x))
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# 编码器单层,多头自注意力层+前馈网络层
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 编码器是把一个x复制三份当做q k v
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 解码器
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory是编码器的q和k
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# 解码器层
class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 从编码来的,q和k
        m = memory
        # 第一个attention，先attention自己
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个attention，
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 前馈网络
        return self.sublayer[2](x, self.feed_forward)


# 掩码层
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# 注意力层,根据q k v计算,返回与输入一样的shape和attention的权重
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #计算权重分数
    if mask is not None:
        # score的shape为(batch,句长,句长) mask的shape为(句长，句长)
        # 在score中，把与之匹配的mask=0的位置变成较少的数
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 多头注意力
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h   # 头数
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model).to(cfg['device']), 4)   #克隆4个 linearLayer
        self.attn = None
        self.dropout = nn.Dropout(p=dropout).to(cfg['device'])

    def forward(self, query, key, value, mask=None):
        # 如果mask不为空
        if mask is not None:
            # 在第二个维度上增加一维
            mask = mask.unsqueeze(1)

        # batchsize
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 前馈网络，两个线性层,d_ff为中间层的神经元个数
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff).to(cfg['device'])
        self.w_2 = nn.Linear(d_ff, d_model).to(cfg['device'])
        self.dropout = nn.Dropout(dropout).to(cfg['device'])

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 嵌入层，根据字典嵌入，注意已经是需要的不要
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 位置编码层，把输入向量进行编码得到位置编码后的向量
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 用了个技巧先计算log的在计算exp
        pe = torch.zeros(max_len, d_model).to(cfg['device'])
        position = torch.arange(0., max_len).unsqueeze(1).to(cfg['device'])
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)).to(cfg['device'])
        # position * div_term 这里生成一个以pos为行坐标，i为列坐标的矩阵
        pe[:, 0::2] = torch.sin(position * div_term).to(cfg['device'])
        pe[:, 1::2] = torch.cos(position * div_term).to(cfg['device'])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x.size(1)就是有多少个pos
        return self.dropout(x)
