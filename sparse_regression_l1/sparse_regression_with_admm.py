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

def get_gauss_kernel(x, c, h):
    return np.exp(-(x-c)**2/2/h**2)

def calc_admm(zk, uk, K_mat, y_vec, lam):
    """反復法により交互方向乗数法の更新式を解く
    Arg:
        zk(ndarray(float)) 前状態のz
        uk(ndarray(float)) 前状態のu
        K_mat(ndarray(ndrray(float))) ガウスカーネル行列←サンプルxに依存
        y_vec(ndarray(float)) サンプルfx
        lam(float) 正則化パラメータ
    Return:
        theta_next(ndarray(float)) 次状態のtheta
        u_next(ndarray(float)) 次状態のu
        z_next(ndarray(float)) 次状態のz
    """
    n = len(y_vec)
    I = np.eye(n)
    theta_next = np.dot(np.linalg.inv((np.dot(K_mat,K_mat)+I)), np.dot(K_mat,y_vec)+zk-uk)
    max_factor = np.vstack((np.zeros(n), theta_next+uk-lam*np.ones(n)))
    min_factor = np.vstack((np.zeros(n), theta_next+uk+lam*np.ones(n)))
    z_next = np.max(max_factor,axis=0) + np.min(min_factor,axis=0)
    u_next = uk+theta_next-z_next
    return theta_next, z_next, u_next

def calc_lasso(X, Y, h, lam):
    """スパース回帰を解く
    Arg:
        X(ndarray(float)) サンプルのx座標の集合
        Y(ndarray(float)) サンプルのy座標の集合
        h(float) ガウス幅
        lam(float) 正則化パラメータ
        z0(float) zの初期状態 default=0
        uo(float) uの初期状態 default=0

    Return:
        x_pred(ndarray(float)) 予測系のx座標←予測しているわけではない
        y_pred(ndarray(float)) 予測されたfx(x座標に対応)
    """
    n = len(Y)
    #初期化
    zk = np.zeros(n)
    uk = np.zeros(n)
    k_matrix = np.empty((n,n))
    #ガウスカーネルマトリックス
    for i in range(len(k_matrix)):
        for j in range(len(k_matrix[0])):
            k_matrix[i][j] = get_gauss_kernel(X[i], X[j], h)
    #ADMMによる更新
    for _ in range(10):
        theta_next, z_next, u_next = calc_admm(zk, uk, k_matrix, Y, lam)
        zk = z_next
        uk = u_next
    #生成フェーズ
    def get_pred_fx(x):
        fx_pred = 0
        for i in range(n):
            k=get_gauss_kernel(x, X[i], h)
            fx_pred += theta_next[i]*k
        return fx_pred
    x_pred = np.linspace(-3, 3, 1000)
    fx_pred = np.empty_like(x_pred)
    for i in range(len(x_pred)):
        fx_pred[i] = get_pred_fx(x_pred[i])

    return x_pred, fx_pred


def plot_graph(x_sample, y_sample, x_pred, y_pred, y_acc):
    plt.clf()
    plt.scatter(x_sample, y_sample, label="sample" ,c='red', marker='o')
    plt.plot(x_pred, y_pred, label="prediction", c='blue')
    plt.plot(x_pred, y_acc, label="true", c="green")
    plt.legend()
    plt.savefig(r"C:\Users\msy-o\Documents\lecture\summer\data_analytics\report\03\output\lasso.png")
    plt.show()
    return None
if __name__ == "__main__":
    x_sample, y_sample = get_sample()
    x_pred, y_pred = calc_lasso(x_sample, y_sample, 0.3, 0.1)
    y_acc = np.empty_like(x_pred)
    for i in range(len(y_acc)):
        y_acc[i] = func(x_pred[i], with_noise=False)
    plot_graph(x_sample, y_sample, x_pred, y_pred, y_acc)