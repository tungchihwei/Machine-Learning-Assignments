import scipy.io
import numpy as np
from numpy import linalg
import itertools
import cvxopt
import cvxopt.solvers

from pandas_ml import ConfusionMatrix

minst = scipy.io.loadmat('MNIST_data.mat')

train_x = np.array(minst["train_samples"])
test_x = np.array(minst["test_samples"])

train_y_tmp = np.array(minst["train_samples_labels"])
train_y = train_y_tmp.reshape((minst["train_samples_labels"].shape[0],))

test_y_tmp = np.array(minst["test_samples_labels"])
test_y = test_y_tmp.reshape((minst["test_samples_labels"].shape[0],))


def cluster(x, y):
    x_clust = [[]]*10

    for i in range(x.shape[0]):
        if not x_clust[y[i]]:
            x_clust[y[i]] = [x[i]]
        else:
            x_clust[y[i]].append(x[i])

    return np.array(x_clust[0]), np.array(x_clust[1]), np.array(x_clust[2]), np.array(x_clust[3]), np.array(x_clust[4]), np.array(x_clust[5]), np.array(x_clust[6]), np.array(x_clust[7]), np.array(x_clust[8]), np.array(x_clust[9])

def polynomial(x, y):
    poly = (1 + np.dot(x, y)) ** 6
    return poly

class SVM():
    def __init__(self):
        self = self

    def train_svm(self, x, y):
        samp, fea = x.shape

        g = np.zeros((samp, samp))
        for i in range(samp):
            for j in range(samp):
                g[i, j] = polynomial(x[i], x[j])

        QP_prob = cvxopt.solvers.qp(
            cvxopt.matrix(np.outer(y,y) * g),
            cvxopt.matrix(np.ones(samp) * (-1)),
            cvxopt.matrix(np.vstack((np.diag(np.ones(samp) * (-1)), np.identity(samp)))),
            cvxopt.matrix(np.hstack((np.zeros(samp), np.ones(samp) * 0.1))),
            cvxopt.matrix(y, (1,samp)),
            cvxopt.matrix(0.0)
            )
        support_v = np.ravel(QP_prob['x']) > 0.00001
        array_i = np.arange(len(np.ravel(QP_prob['x'])))[support_v]
        self.rav = np.ravel(QP_prob['x'])[support_v]
        self.x_support_v = x[support_v]
        self.y_support_v = y[support_v]

        self.inter = 0
        for i in range(len(self.rav)):
            self.inter = (self.inter + self.y_support_v[i] - np.sum(self.rav * self.y_support_v * g[array_i[i],support_v])) / len(self.rav)

    def comp(self,x):
        y_pre = np.zeros(len(x))
        for i in range(len(x)):
            tmp = 0
            z = zip(self.rav, self.y_support_v, self.x_support_v)
            for rav, y_support_v, x_support_v in z:
                # t = polynomial(x[i], x_support_v)
                tmp += rav * y_support_v * polynomial(x[i], x_support_v)
            # print(tmp)

            y_pre[i] = tmp
        result = y_pre + self.inter
        
        return result


def tree(label, dic):
    combine = list(itertools.combinations(label, 2))
    dt = None
    l = label
    if len(combine) == 1:
        if dic[combine[0]] >= 0:
            dt = combine[0][0]
        else:
            dt = combine[0][1]
    elif len(combine) > 1:
        if dic[combine[0]] >= 0:
            # tmp = l[1]
            l.pop(1)
            dt = tree(l, dic)
        else:
            # tmp = l[0]
            l.pop(0)
            dt = tree(l, dic)
    return dt

x_trains_pre = [[]]*10
x_trains_pre[0:10] = cluster(train_x, train_y)

list_pre = []

for i in list(itertools.combinations([0,1,2,3,4,5,6,7,8,9], 2)):
    y_1 = np.ones(len(x_trains_pre[i[0]]))
    y_2 = np.ones(len(x_trains_pre[i[1]])) * (-1)
    train = np.vstack((x_trains_pre[i[0]] , x_trains_pre[i[1]]))
    test = np.hstack((y_1, y_2))
    svm = SVM()
    svm.train_svm(train, test)
    pre_y = svm.comp(test_x)
    list_pre.append(pre_y)

pre = []
label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
label_list = list(itertools.combinations(label, 2))

for i in np.transpose(np.array(list_pre)):
    dic = {}
    for j in range(len(label_list)):
        dic[label_list[j]] = i[j]
    dtree = tree([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dic)
    pre.append(dtree)




confusion = ConfusionMatrix(test_y, np.array(pre))
accuracy = (np.sum(np.array(pre) == test_y) / float(len(pre_y))) * 100

print("DAGSVM confusion matrix: ")
print(confusion)
print("DAGSVM accuracy: ", accuracy, "%")





