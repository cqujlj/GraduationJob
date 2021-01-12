import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os

# RMSE
def draw_RMSE_and_MAE(data_path):
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 获得数据集
    RMSE = np.loadtxt(os.path.join(data_path,"RMSE.csv"))
    MAE = np.loadtxt(os.path.join(data_path, "MAE.csv"))


    # 对比最优值
    #trraffic: RMSE:0.03853-0.04245  MAE:0.02664-0.02974
    #electricity: RMSE:73.42-83.34  MAE:56.91- 64.66
    #power: RMSE:0.4963- 0.9669  MAE:0.2732 - 52.61
    RMSE_best_line, = plt.plot(np.array([0,MAE.shape[0]-1]), np.array([83.34,83.34]), 'r--')
    MAE_best_line, = plt.plot(np.array([0,MAE.shape[0]-1]), np.array([64.66,64.66]), 'b--')


    # 创建x
    x = np.arange(1, RMSE.shape[0] + 1, 1)

    RMSE_line, = plt.plot(x, RMSE, 'r-')
    MAE_line, = plt.plot(x, MAE, 'b-')

    plt.legend(handles=[RMSE_line,MAE_line,RMSE_best_line,MAE_best_line],
               labels=["RMSE","MAE","best_RMSE","best_MAE"],
               loc="upper right",
               fontsize=10)

    # 填充最大值最小值
    plt.xlabel("epochs")
    y_major_locator = MultipleLocator(20)  # 设置y轴间隔
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 200)
    plt.ylabel("RMSE")
    plt.grid(linestyle='--', color='gray', )
    plt.show()


# 主函数
def main():
    draw_RMSE_and_MAE("F:/20210107-实验记录")


if __name__ == '__main__':
    main()
