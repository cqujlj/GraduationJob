# 配置文件
config = {
    # --------------------------transformer相关------------------------
    # 输入x的嵌入维度
    'd_model': 960,
    # 前馈网络中间层神经元的个数
    'd_ff': 2048,
    # 多头注意力的头个数
    'head': 15,
    # 编码器编码几个编码单元
    'encoder_N': 6,
    # 解码器编码几个编码单元
    'decoder_N': 6,
    # dropout死掉的神经元的几率
    'dropout': 0.1,

    # ------------------------------------LSTM相关-----------------------------
    'hidden_size': 2048,
    'num_layers': 1,
    'LSTM_dropout': 0.1,

    # ----------------------------环境相关----------------
    'device': 'cuda',

    # ---------------------------数据集环境--------------
    # 'data_path': "/home/lwj/datasets/timeseries/datasets/use/power",# 本地
    'data_path': "/home/mlg1504/lwj/timeseries/tran",  # 服务器
    'time_step': 3,
    'normalize': 1.0,

    # -----------------------------
    'epochs': 20000,
    'train_batch_size': 64,
    'test_batch_size': 64,
}
