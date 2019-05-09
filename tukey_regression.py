## Requirement
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(114514)

def get_sample(x_min=-3., x_max=3., sample_num=10, theta_1=0., theta_2=1.):
    """サンプルを生成する。
    講義資料にあるように
        y = theta_1 + theta_2 x
    から点を作る。
    この時，ノイズを生成させる。
    ただし，外れ値を作る必要があるので，数点ぶっ飛ばす。。。

    Arg:
        x_min(float) x座標の最小点 default=-3.
        x_max(float) x座標の最大点 default=3.
        sample_num(int) (x,y)の生成個数 default=10
        theta_1(float) theta_1(y切片) defalut=0.
        theta_2(float) theta_2(傾き) default=1.

    Return:
         X, Y(tuple(list(float),list(float))) サンプル(x,y)の
    """
    X = np.linspace(x_min, x_max, sample_num)
    # ノイズを載せてy座標をサンプル
    Y = theta_1 + theta_2*X + np.random.normal(loc=0, scale=0.2, size=sample_num)
    # 外れ値を設定
    Y[-1] = Y[-2] = Y[1] = -4
    return X, Y

def tukey_weight(r, eta):
    """テューキー損失に対する重み
    Arg:
        r(float) 予測値とサンプルyの残差

    Return:
        w(float) 重み
    """
    if abs(r) <= eta:
        return (1 - r**2/eta**2)**2
    else:
        return 0

def get_basal(x):
    """今回の基底関数の定義
    Arg:
        x
    Return:
        phi_1, phi_2 = 1, x
    """
    return np.array([1, x])

def calc_tukey_regression(X, Y, theta_init=[1.,1.], eta=1., n_iter_max=1000):
    """テューキー回帰を解く。
    thetaの初期値によっては，非凸性により悲しみを帯びるので注意。
    Arg:
        X(ndarray(float)) サンプルのx座標
        Y(ndarray(float)) サンプルのy座標
        theta_init(list[float, float]) thetaの初期値←←重要
        eta(float) 外れ値をテキトーに除外してくれるパラメータ default=1.
        n_iter_max(int) イテレーションの最大数 default=1000

    """
    n = len(X) #サンプル数
    b = len(get_basal(X[0])) #基底関数数
    # 計画行列
    Phi_mat = np.empty((n, b))
    for row in range(n):
        Phi_mat[row] = get_basal(X[row])
     #thetaの初期値
    theta_vec = np.array(theta_init)
    # 以下繰り返し再重みづけ最小二乗
    for _ in range(n_iter_max):
        r = np.abs(np.dot(Phi_mat, theta_vec)-Y)
        #対角成分を抽出
        w_array = np.array([tukey_weight(r_i,eta) for r_i in r])
        W = np.diag(w_array)
        phit_w_phi = Phi_mat.T.dot(W).dot(Phi_mat)
        phit_w_y = Phi_mat.T.dot(W).dot(Y)
        theta_vec_pred = np.linalg.solve(phit_w_phi,phit_w_y)
        if np.linalg.norm(theta_vec_pred-theta_vec) < 1e-4:
            theta_vec = theta_vec_pred
            break
        else:
            theta_vec = theta_vec_pred
    return theta_vec

if __name__ == "__main__":
    X, Y = get_sample()
    theta_vec = calc_tukey_regression(X,Y)
    X_detail = np.linspace(-3., 3., 100)
    Y_detail = np.empty_like(X_detail)
    for i in range(len(X_detail)):
        Y_detail[i] = get_basal(X_detail[i]).dot(theta_vec)

    sample_filename = "sample.png"
    plt.scatter(X, Y, c="blue", marker="o", label="sample")
    plt.legend()
    plt.savefig(sample_filename)
    plt.show()

    filename = "tukey_output.png"
    plt.plot(X_detail, Y_detail, color="red", label="rediction")
    plt.scatter(X, Y, c="blue", marker="o", label="sample")
    plt.legend()
    plt.savefig(filename)
    plt.show()