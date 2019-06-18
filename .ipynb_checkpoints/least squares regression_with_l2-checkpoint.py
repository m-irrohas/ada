##Requirement
import numpy as np
import matplotlib.pyplot as plt
import random

def func(x, with_noise=True):
    """基本方程式
    Arg:
        x(float) x座標
        with_noise(boolian) ノイズ入れるかどうか
    Return:
        fx(float) x座標に対応するy(with_noise?)
    """
    noise = 0.05 * np.random.normal(loc=0., scale=1.) if with_noise else 0
    return np.sin(np.pi*x)/np.pi/x+0.1*x + noise

def get_sample(x_min=-3., x_max=3., num_sample=50):
    """サンプル生成
    Arg:
        x_min(float) x座標の最小値 default=-3.
        x_max(float) x座標の最大値 default=3.
        num_sample(int) サンプルの生成数(=dim(theta)) default=50

    Return:
        x_lst(list(float)) x座標の集合
        fx_lst(list(float)) y座標の集合
    """
    x_lst = np.linspace(x_min, x_max, num_sample)
    fx_lst = np.array([func(x, with_noise=True) for x in x_lst])
    return x_lst, fx_lst

def calc_least_squared_parameter(x_lst, fx_lst, h, lam, num=1000):
    """l2正則化付きで最小二乗回帰を解く
    Arg:
        x_lst(ndarray(float)) サンプルのx座標
        fx_lst(ndarray(float)) サンプルの出力系(xに対応)
        h(float) ハイパーパラメータ

    Return:
        x_lst_detail(ndarray(float)) 予測系のx座標(自分で設定する)
        fx_pred(ndarray(float)) 最小二乗回帰で予測されたf(x)(xに対応させて。)
    """
    #ガウスカーネル
    def get_kernel(x, c, h):
        return np.exp(-(x-c)**2/2/h**2)
    #次元
    n = len(x_lst)
    k_matrix = np.empty((n,n))
    for i in range(len(k_matrix)):
        for j in range(len(k_matrix[0])):
            k_matrix[i][j] = get_kernel(x_lst[i], x_lst[j], h)
    #theta(dim=n)
    theta_pred = np.dot(np.linalg.inv((np.dot(k_matrix, k_matrix)+lam*np.eye(n))), np.dot(k_matrix.T, fx_lst))
    #以下生成フェーズ
    x_lst_detail = np.linspace(-3, 3, num=num)
    #fx値の予測(引数x)
    def get_pred_fx(x):
        fx_pred = 0
        for i in range(n):
            k=get_kernel(x, x_lst[i], h)
            fx_pred += theta_pred[i]*k
        return fx_pred

    fx_pred_lst = np.empty_like(x_lst_detail)
    for i in range(len(x_lst_detail)):
        fx_pred_lst[i] = get_pred_fx(x_lst_detail[i])

    return x_lst_detail, fx_pred_lst

def calc_least_squared_parameter_with_cross_valid(x_lst, fx_lst, h, lam, k_num):
    """クロスバリデーション付きでl2最小二乗回帰を解く
    Arg:
        x_lst(ndarray(float))サンプルのx座標
        fx_lst(ndarray(float))サンプルの出力座標(x座標に対応)
        h(float)
        lam(float)
        k_num(int) 分割数

    Return:
        err_mean(float) 誤差値の平均
    """
    #サンプル数
    sample_num = len(x_lst)
    #index→シャッフルさせる
    idx = list(range(sample_num))
    random.shuffle(idx)
    #分割された時の1リスト当たりの個数
    split_lst_num = int(sample_num/k_num)
    #indexを分割
    idx_split = [idx[i:i+split_lst_num] for i in range(0, sample_num, split_lst_num)]
    err = 0
    for ki in range(k_num):
        test_idx = idx_split[ki]
        train_idx = []
        for idx in idx_split:
            if idx == test_idx:
                pass
            else:
                for i in idx:
                    train_idx.append(i)
        train_x = [x_lst[i] for i in train_idx]
        train_y = [fx_lst[i] for i in train_idx]
        test_x = [x_lst[i] for i in test_idx]
        test_y = [fx_lst[i] for i in test_idx]
        #最小二乗回帰により解く
        _, y_pred = calc_least_squared_parameter(train_x, train_y, h, lam, num=sample_num)
        y_pred_for_err = [y_pred[i] for i in test_idx]
        for i in range(len(test_x)):
            err += (y_pred_for_err[i]-test_y[i])**2
    err /= k_num
    return err

def plot_graph(x_sample, y_sample, x_pred, y_pred, y_acc):
    fig = plt.figure()
    plt.clf()
    plt.scatter(x_sample, y_sample, label="sample" ,c='red', marker='o')
    plt.plot(x_pred, y_pred, label="prediction", c='blue')
    plt.plot(x_pred, y_acc, label="true", c="green")
    plt.legend()
    plt.savefig(r"C:\Users\msy-o\Documents\lecture\summer\data_analytics\report\02\output\prediction.png")
    plt.show()
    return None

#サンプル生成(とりあえず50個)
x_sample, y_sample = get_sample()
#ガウス幅h
gauss_widths = [0.1, 0.3, 1]
#正則化パラメータ
regularize_params = [0.05, 0.1, 0.5]
#分割数k
k = 10

errs = []
#hとラムダを変化させて実験
for h in gauss_widths:
    for l in regularize_params:
        err = calc_least_squared_parameter_with_cross_valid(x_sample, y_sample, h, l, k)
        print("ガウス幅:%1.2f 正則化パラム:%1.2f err:%1.4f" %(h, l, err))
        errs.append([h,l,err])
#最も評価の良かったhと正則化パラムとその時のerrをとる
err_min = min(errs, key=lambda x: x[2])
print(err_min)
x_pred, y_pred = calc_least_squared_parameter(x_sample, y_sample, h=err_min[0], lam=err_min[1])
y_acc = np.empty_like(x_pred)
for i in range(len(y_acc)):
    y_acc[i] = func(x_pred[i], with_noise=False)

plot_graph(x_sample, y_sample, x_pred, y_pred, y_acc)
