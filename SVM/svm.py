import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size):
    """Generate training data.

    Since
    f(x) = w^{T}x + b
    can be written as
    f(x) = (w^{T}, b)(x^{T}, 1)^{T},
    for the sake of simpler implementation of SVM,
    we return (x^{T}, 1)^{T} instead of x

    :param sample_size: number of data points in the sample
    :return: a tuple of data point and label
    """

    x = np.random.normal(size=(sample_size, 3))
    x[:, 2] = 1.
    x[:sample_size // 2, 0] -= 5.
    x[sample_size // 2:, 0] += 5.
    y = np.concatenate([np.ones(sample_size // 2, dtype=np.int64),
                        -np.ones(sample_size // 2, dtype=np.int64)])
    x[:3, 1] -= 5.
    y[:3] = -1
    x[-3:, 1] += 5.
    y[-3:] = 1
    return x, y


def svm(x, y, l, lr):
    """Linear SVM implementation using gradient descent algorithm.

    f_w(x) = w^{T} (x^{T}, 1)^{T}

    :param x: data points shape=(sample_size, 3)
    :param y: label shape=(sample_size, 1)
    :param l: regularization parameter
    :param lr: learning rate
    :return: three-dimensional vector w
    """
    def get_subgradient(w, x, y):
        """劣勾配を求める。
        Arg:
            w(ndarray(float)):重さ, shape=(3)
            x(ndarray(float)):サンプルデータ, shape=(-1, 3)
            y(ndarray(int)):ラベルデータ, shape(-1)
        Return
            delta(ndarray(float)):劣勾配, shape=(-1, 3)
        """
        sub_delta = np.zeros_like(x)
        loss = 1-np.dot(x, w)*y #shape=(-1)
        for i in range(len(loss)):
            sub_delta[i, :] = -y[i]*x[i,:] if loss[i] >0 else 0
        #delta = np.array([-y[i]*x[i,:] if loss[i] > 0 else pass for i in range(len(loss))])
        return sub_delta

    w = np.zeros(3)
    prev_w = w.copy()
    for i in range(10 ** 4):
        w -= lr*(np.sum(get_subgradient(w, x, y), axis=0) + l*np.eye(x.shape[1]).dot(w))
        if np.linalg.norm(w - prev_w) < 1e-3:
            break
        prev_w = w.copy()
    if i%100==0:
        print("%f is over"% i)
    return w


def visualize(x, y, w):
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label="calss:1")
    plt.scatter(x[y == -1, 0], x[y == -1, 1], label="class:2")
    plt.plot([-10, 10], -(w[2] + np.array([-10, 10]) * w[0]) / w[1], label="boundary by SVM")
    plt.legend()
    plt.savefig('lecture6-h2.png')


x, y = generate_data(200)
w = svm(x, y, l=.1, lr=1.)
visualize(x, y, w)

plt.clf()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter(x[y == 1, 0], x[y == 1, 1], label="class:1")
plt.scatter(x[y == -1, 0], x[y == -1, 1], label="class:2")
plt.legend()
plt.savefig("sample.png")