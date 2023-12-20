import numpy as np
import sklearn.datasets
import urllib.request
from optmethods.loss import LogisticRegression


class W8a_dataset:
    def __init__(self, num_cpus):
        w8a_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
        data_path = './w8a'

        f = urllib.request.urlretrieve(w8a_url, data_path)
        A, b = sklearn.datasets.load_svmlight_file(data_path)

        n, dim = A.shape
        if n % num_cpus != 0:
            A = A[:n - (n % num_cpus)]
            b = b[:n - (n % num_cpus)]
        b_unique = np.unique(b)
        if (b_unique == [1, 2]).all():
            # Transform labels {1, 2} to {0, 1}
            b = b - 1
        elif (b_unique == [-1, 1]).all():
            # Transform labels {-1, 1} to {0, 1}
            b = (b+1) / 2
        else:
            # replace class labels with 0's and 1's
            b = 1. * (b == b[0])

        self.A = A
        self.b = b

        loss = LogisticRegression(self.A, self.b, l1=0, l2=0)
        self.n, self.dim = A.shape
        if self.n <= 20000 or self.dim <= 20000:
            print('Computing the smoothness constant via SVD, it may take a few minutes...')
        self.L = loss.smoothness
        self.l2 = 1e-4 * self.L
        print(f'L: {self.L}, l2: {self.l2}')
        
        
    def get_full_data(self):
        loss = LogisticRegression(self.A, self.b, l1=0, l2=self.l2)
        x0 = np.zeros(self.dim)
        n_epoch = 100
        trace_len = 300

        return loss, x0, n_epoch, trace_len

    def get_splitted_data(self, n_workers):
        permutation = self.b.squeeze().argsort()

        losses = []
        idx = [0] + [(self.n * i) // n_workers for i in range(1, n_workers)] + [self.n]
        for i in range(n_workers):
            idx_i = permutation[idx[i] : idx[i+1]]
            # idx_i = range(idx[i], idx[i + 1])
            loss_i = LogisticRegression(self.A[idx_i].A, self.b[idx_i], l1=0, l2=self.l2)
            loss_i.computed_grads = 0
            losses.append(loss_i)

        return losses    
