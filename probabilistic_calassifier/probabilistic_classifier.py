import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size=90, n_class=3):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y


def solve_LSPC(x, y, width, n_class=3, regurater=10.):
     """最小二乗確率的分類を解く
     Arg:
          x(ndarray(float)) サンプルのx座標 shape=90
          y(ndarray(int)) クラス(0,1,2)のどれか shape=90
          width(float) ガウス幅
          n_class(int) クラス数←既知として進める。
          ragurater(float) 正則化係数

     Return:
          theta(ndarray) 分類結果の係数行列
     """
     sample_size = len(x)
     Phi = np.exp(-(x[:] - x[:,None])**2 / (2 * width ** 2))
     y0 = np.where(y==0, 1, 0)
     y1 = np.where(y==1, 1, 0)
     y2 = np.where(y==2, 1, 0)
     Pi = np.array([y0,y1,y2]).T
     Theta = np.linalg.solve((Phi.T.dot(Phi)+regurater*np.eye(sample_size)), Phi.T.dot(Pi))
     return Theta

x,y = generate_data()
plt.clf()
plt.scatter(x[y==0], y[y==0], c='blue', marker="o", label="class:0")
plt.scatter(x[y==1], y[y==1], c='red', marker="x", label="class:1")
plt.scatter(x[y==2], y[y==2], c='green', marker="v", label="class:2")
ylabels=[int(0),int(1),int(2)]
plt.yticks(y, ylabels)
plt.legend()
plt.savefig("train_sample.png")
plt.show()

h = 1.
theta = solve_LSPC(x, y, h)

X = np.linspace(-5., 5., num=100)
K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))
plt.clf()
plt.xlim(-5, 5)
plt.ylim(-.3, 1.5)
logit = K.dot(theta)
print(logit.shape)
zeros_mat = np.zeros_like(logit)
unnormalized_prob = np.maximum(zeros_mat, logit)
prob = unnormalized_prob / unnormalized_prob.sum(1, keepdims=True)

plt.plot(X, prob[:, 0], c='blue', label="probability-class: 1")
plt.plot(X, prob[:, 1], c='red' ,label="probability-class: 2")
plt.plot(X, prob[:, 2], c='green', label="probability-class: 3")
plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')
plt.legend()
plt.savefig("LSPC.png")
plt.show()