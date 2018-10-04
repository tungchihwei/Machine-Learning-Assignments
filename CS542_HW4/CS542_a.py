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

accuracy_one = 0
confusion_one = None


def cluster(x, y):
    x_clust = [[]]*10

    for i in range(x.shape[0]):
        if not x_clust[y[i]]:
            x_clust[y[i]] = [x[i]]
        else:
            x_clust[y[i]].append(x[i])

    return np.array(x_clust[0]), np.array(x_clust[1]), np.array(x_clust[2]), np.array(x_clust[3]), np.array(x_clust[4]), np.array(x_clust[5]), np.array(x_clust[6]), np.array(x_clust[7]), np.array(x_clust[8]), np.array(x_clust[9])

def join_func(x, val):
	join = []
	for i in x:
		join.append(i)

	# print("##############################")
	# print(len(join))
	# print("##############################")

	tmp = join[val]
	join[val] = join[1]
	join.pop(1)

	# train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10 = x[0:9]

	x_all = np.vstack(join[0:len(join)])
	y_val = np.ones(len(tmp))
	y_all = np.ones(len(x_all)) * (-1)

	return tmp, x_all, y_val, y_all

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
    def pre(self, x):
        pre = np.sign(self.comp(x))
        return pre


def one():
    x_trains_pre = [[]]*10
    x_trains_pre[0:10] = cluster(train_x, train_y)
    # print("0: ", len(x_trains_pre[0]))
    # print(len(x_trains_predict))
    # print(trainX)
    list_trans = []
    
    for i in list(itertools.combinations([0,1,2,3,4,5,6,7,8,9], 2)):
        # print(i[0])
        # print(i[1])
        # print("i end")
        y_1 = np.ones(len(x_trains_pre[i[0]]))
        y_2 = np.ones(len(x_trains_pre[i[1]])) * (-1)
        # print(len(x_trains_pre[i[0]]))
        # print(len(x_trains_pre[i[1]]))
        # print("end")
        # print(len(y1))
        # print(len(y2))
        # break
        train = np.vstack((x_trains_pre[i[0]] , x_trains_pre[i[1]]))
        test = np.hstack((y_1, y_2))

        svm = SVM()
        svm.train_svm(train, test)

        pre_y = svm.pre(test_x)

        for j in range(pre_y.shape[0]):
            if pre_y[j] == -1:
                pre_y[j] = i[1]
            elif pre_y[j] == 1:
                pre_y[j] = i[0]
        list_trans.append(pre_y)

    list_trans = np.array(list_trans).astype(int)
    trans = np.transpose(list_trans)

    predicted = []


    for i in range(trans.shape[0]):
        predicted.append(np.argmax(np.bincount(trans[i])))


    confusion = ConfusionMatrix(test_y, np.array(predicted))
    
    accuracy = (np.sum(np.array(predicted) == test_y) / float(len(pre_y))) * 100

    return confusion, accuracy


    # print("confusion matrix: ")
    # print(confusion_one)
    # print("accuracy: ", accuracy_one, "%")

    
accuracy_all = 0
confusion_all = None
def all():
	x_trains_pre = [[]]*10
	x_trains_pre[0:10] = cluster(train_x, train_y)

	

	list_pre = []

	for i in range(10):
		# join(x_trains_pre, i)
		# break
		train_n, train_all, test_n, test_all = join_func(x_trains_pre, i)

		train = np.vstack((train_n,train_all))
		test = np.hstack((test_n, test_all))

		svm = SVM()
		svm.train_svm(train, test)
		pre_y = svm.comp(test_x)
		list_pre.append(pre_y)

	pre = np.array(list_pre)

	predict = np.argmax(pre, axis = 0)

	confusion = ConfusionMatrix(test_y, predict)
	accuracy = (np.sum(predict == test_y) / float(len(pre_y))) * 100

	return confusion, accuracy


confusion_one, accuracy_one = one()
confusion_all, accuracy_all = all()

print("one versus one confusion matrix: ")
print(confusion_one)
print("one versus one accuracy: ", accuracy_one, "%")

print("one versus the rest confusion matrix: ")
print(confusion_all)
print("one versus the rest accuracy: ", accuracy_all, "%")










