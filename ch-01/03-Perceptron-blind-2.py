'''
Notes


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(x=xi))
                self.w_[0] += update
                self.w_[1:] += xi * update # missed a += here
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def predict(self, x):
        return np.where(self.net_input(x=x) >= 0.0, 1, -1)

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)
plt.figure()
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)
plt.show()
