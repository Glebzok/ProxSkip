import numpy as np
import ray


@ray.remote
class Worker:
    def __init__(self, shuffle=True, prox_skip=False, loss=None, it_local=None, batch_size=1):
        self.loss = loss
        self.shuffle = shuffle
        self.prox_skip = prox_skip
        self.it_local = it_local
        self.batch_size = batch_size
        self.c = None
        self.h = None
        self.rng_skip = np.random.default_rng(42) # random number generator for random synchronizations
    
    def run_local(self, x, lr):
        self.x = x * 1.
        if self.shuffle:
            self.run_local_shuffle(lr)
        elif self.prox_skip:
            self.run_prox_skip(lr)
        else:
            self.run_local_sgd(lr)
        return self.x
    
    def run_local_shuffle(self, lr):
        permutation = np.random.permutation(self.loss.n)
        i = 0
        while i < self.loss.n:
            i_max = min(self.loss.n, i + self.batch_size)
            idx = permutation[i:i_max]
            self.x -= lr * self.loss.stochastic_gradient(self.x, idx=idx)
            i += self.batch_size
    
    def run_local_sgd(self, lr):
        for i in range(self.it_local):
            if self.batch_size is None:
                self.x -= lr * self.loss.gradient(self.x)
            else:
                self.x -= lr * self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
    
    def run_scaffold(self, x, lr, c):
        # as in the original scaffold paper, we use their Option II
        self.x = x * 1.
        if self.c is None:
            self.c = self.x * 0. #initialize zero vector of the same dimension
        for i in range(self.it_local):
            if self.batch_size is None:
                g = self.loss.gradient(self.x)
            else:
                g = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (g - self.c + c)
        self.c += 1 / (self.it_local * lr) * (x - self.x) - c
        return self.x

    def run_prox_skip(self, lr):
        p = 1 / self.it_local
        if self.h is None:
            # first iteration
            self.h = self.x * 0. # initialize zero vector of the same dimension
        else:
            # update the gradient estimate 
            self.h += p / lr * (self.x - self.x_before_averaing)
        it_local = self.rng_skip.geometric(p=p) # since all workers use the same random seed, this number is the same for all of them

        for i in range(it_local):
            if self.batch_size is None:
                g = self.loss.gradient(self.x)
            else:
                g = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.
    
    def run_fedlin(self, x, lr, g):
        self.x = x * 1.
        for i in range(self.it_local):
            if self.batch_size is None:
                grad = self.loss.gradient(self.x)
            else:
                grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (grad - self.g + g)
        return self.x
    
    def run_slocalgd(self, x, lr, g):
        self.x = x * 1.
        p = 1 / self.it_local
        it_local = self.rng_skip.geometric(p=p)
        for i in range(self.it_local):
            if self.batch_size is None:
                grad = self.loss.gradient(self.x)
            else:
                grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (grad - self.g + g)
        return self.x
    
    def get_control_var(self):
        return self.c
    
    def get_h(self):
        return self.h
    
    def get_g(self):
        return self.g
    
    def get_fedlin_grad(self, x):
        if self.batch_size is None:
            self.g = self.loss.gradient(x)
        else:
            self.g = self.loss.stochastic_gradient(x, batch_size=self.batch_size)
        return self.g
