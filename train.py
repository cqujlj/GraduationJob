from torch import nn
from torchsummaryX import summary
from load_data import *
from models import LSTM,My_transformer


# 主函数
def main():
    # 加载训练集和测试集以及掩码矩阵
    train_dataset = MyDataset("train")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['train_batch_size'],
                              shuffle=True)

    test_dataset = MyDataset("test")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg['test_batch_size'],
                             shuffle=False)

    tgt_mask = get_mask(cfg['time_step']).to(cfg['device'])

    # 加载模型到设备
    model = My_transformer()
    # model = LSTM()
    x1 = torch.ones(cfg['train_batch_size'], 12, cfg['d_model']).to(cfg['device'])  # 1-12
    x2 = torch.ones(cfg['train_batch_size'], cfg['time_step'], cfg['d_model']).to(cfg['device'])  # 12-14
    summary(model, x1, x2, None, None)
    model.to(cfg['device'])

    # 模型初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 损失函数和评价指标
    MSE = nn.MSELoss().to(cfg['device'])
    MAE = nn.L1Loss().to(cfg['device'])

    # 优化器
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.RMSprop(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters())

    # 测试集的指标
    test_RMSE_all = []
    test_MAE_all = []
    for epoch in range(cfg['epochs']):
        # 损失函数
        train_loss = 0
        train_RMSE_loss = 0
        train_MAE_loss = 0

        for step, batch_data in enumerate(train_loader):
            # 每一步训练
            model.train()
            # 获得数据
            x1, x2, label = batch_data
            # 前向传播
            pre = model(x1, x2, None, tgt_mask)
            # 计算损失
            loss = MSE(pre , label)
            # 因为每次反向传播的时候，变量里面的梯度都要清零
            optimizer.zero_grad()
            # 变量得到了grad
            loss.backward()
            # 更新参数
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_RMSE_loss += np.sqrt(loss.item())
            train_MAE_loss += MAE(pre, label).item()
            # 统计
            if step % 20 == 0:
                print("Epoch:%d   step:%d   loss:%f   RMSE:%f   MAE:%f" % (epoch,
                                                                           step,
                                                                           train_loss / (step + 1),
                                                                           train_RMSE_loss / (step + 1),
                                                                           train_MAE_loss / (step + 1)))

            # 每200轮测试一下
            if step % 200 == 0:
                # 每一轮完了之后跑测试集合
                test_RMSE = []  # MSE开根号
                test_MAE = []
                model.eval()
                with torch.no_grad():
                    for step, batch_data in enumerate(test_loader):
                        # 获得数据
                        x1, x2, label = batch_data

                        # 直接预测
                        # pre = model(x1, x2, None, tgt_mask)

                        # 预测
                        feed = x2[:, 0:1, :]  # 第一个
                        for time in range(3):
                            # 预测
                            pre = model(x1, feed, None, None)
                            # 拿到第13+time的值
                            if time == 0:
                                pres = pre[:, time:time + 1, :]
                            else:
                                pres = torch.cat((pres, pre[:, time:time + 1, :]),dim=1)
                            # 并且把这一天的值作为新的输入
                            feed = torch.cat((feed, pre[:, time:time + 1, :] / cfg['normalize']), dim=1)

                        # 统计
                        test_RMSE.append(np.sqrt(MSE(pres, label).item()))
                        test_MAE.append(MAE(pres, label).item())

                # 统计完成
                RMSE_num = np.array(test_RMSE).mean()
                MAE_num = np.array(test_MAE).mean()

                print("Test: Epoch %d RMSE %f MAE %f" % (epoch, RMSE_num, MAE_num))
                print()

                # 保存到磁盘
                test_RMSE_all.append(RMSE_num)
                test_MAE_all.append(MAE_num)
                np.savetxt("RMSE.csv", np.array(test_RMSE_all))
                np.savetxt("MAE.csv", np.array(test_MAE_all))


if __name__ == "__main__":
    main()
