#Requirement
import numpy as np
import matplotlib.pyplot as plt
import random
import os

np.random.seed(1)  # set the random seed for reproducibility

#出力のディレクトリの作成
if not os.path.exists("output"):
    os.mkdir("output")

def data_generate(n=50):
    """訓練データ生成
    Arg:
        n(int) 生成データ数 default=50
    
    Return:
        x(ndarray(float)) 生成データの入力
            shape=(n,3)
            x_i = [x1, x2, 1]で構成される
        
        y(ndarray(int)) 生成データのラベル
            shape=(n,)
            y_i = {-1, 1}
    """
    x = np.random.randn(n, 3)
    x[:n // 2, 0] -= 15
    x[n // 2:, 0] -= 5
    x[1:3, 0] += 10
    x[:, 2] = 1
    y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    index = np.random.permutation(np.arange(n))
    return x[index], y[index]

def AdaptedRegulateClassifer(x, y):
    """適応正則化分類
    Arg:
        x(ndarray(float)) 生成データの入力
            shape=(n,3)
            x_i = [x1, x2, 1]で構成される

        y(ndarray(int)) 生成データのラベル
            shape=(n,)
            y_i = {-1, 1}

    Return:
        mu(ndarray(float)) 適応正則化によるmu

    Memo:
        phi = (x1, x2, 1).T
        muとSigmaを求める
        反復法
    """
    iteration=1000
    gamma = 10
    indexes = range(50)
    mu = np.array([1,1,1]).reshape(3,1)
    print("mu shape : {}".format(mu.shape))
    print(mu)
    Sigma = np.eye(3)
    print("Sigma shape : {}".format(Sigma.shape))
    print(Sigma)
    for index in range(iteration):
        index = np.random.choice(indexes)
        index = index%50
        y_i = y[index]
        phi = x[index].reshape(3,1)
        beta = np.dot(phi.T, Sigma.dot(phi))+gamma
        phiphiT = np.dot(phi, phi.T)
        Sigma_next = Sigma - np.dot(Sigma.dot(phiphiT),Sigma)/float(beta)
        mu_next = mu + y_i*max(0, 1-y_i*mu.T.dot(phi))/beta*Sigma.dot(phi)
        mu = mu_next
        Sigma = Sigma_next
    return mu

x, y = data_generate()
class0_index = np.where(y==-1)
class1_index = np.where(y==1)
class0_x = x[class0_index]
class1_x = x[class1_index]
class0_x = np.delete(class0_x, 2, axis=1)
class1_x = np.delete(class1_x, 2, axis=1)
#サンプルプロット
plt.clf()
plt.scatter(class0_x[:,0], class0_x[:,1], marker="o", label="class:0")
plt.scatter(class1_x[:,0], class1_x[:,1], marker="v", label="class:1")
plt.ylim(-2,2.4)
plt.xlim(-18,-2)
plt.legend()
plt.savefig("./output/sample_plot.png")
plt.show()

plt.clf()
plt.scatter(class0_x[:,0], class0_x[:,1], marker="o", label="class:0")
plt.scatter(class1_x[:,0], class1_x[:,1], marker="v", label="class:1")
mu = AdaptedRegulateClassifer(x, y)
x0 = np.linspace(-12,-8, num=50)
x1 = -mu[0]/mu[1]*x0-mu[2]/mu[1]
plt.plot(x0,x1, label="boundary")
plt.ylim(-2,2.4)
plt.xlim(-18,-2)
plt.legend()
plt.savefig("./output/result.png")
plt.show()
