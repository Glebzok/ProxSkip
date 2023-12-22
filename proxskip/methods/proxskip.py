import copy
import numpy as np
import ray
import psutil

from optmethods.optimizer import StochasticOptimizer
from methods.utils import Worker


class ProxSkip(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        lr_decay_coef (float, optional): the coefficient in front of the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/2, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        it_start_decay (int, optional): how many iterations the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
    """
    def __init__(self, it_local, n_workers=None, cohort_size=None, iid=False, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, it_start_decay=None, batch_size=1, losses=None, *args, **kwargs):
        super(ProxSkip, self).__init__(*args, **kwargs)
        self.it_local = it_local
        if n_workers is None:
            n_workers = psutil.cpu_count(logical=False)
        if cohort_size is None:
            cohort_size = n_workers
        self.n_workers = n_workers
        self.cohort_size = cohort_size
        self.iid = iid
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        self.batch_size = batch_size
        self.losses = losses
        self.trace.hs = []
        
        

             
    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local*self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)
        
        if self.cohort_size == self.n_workers:
            self.x = np.mean(ray.get([worker.run_local.remote(x_id, self.lr, proba=self.proba) for worker in self.workers]), axis=0)
            self.h = ray.get([worker.get_h.remote() for worker in self.workers])       
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            self.x = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr) for i in cohort]), axis=0)
            self.h = ray.get([worker.get_h.remote() for worker in self.workers])       
    
    def init_run(self, *args, **kwargs):
        super(ProxSkip, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [Worker.remote(shuffle=False, prox_skip=True, loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [Worker.remote(shuffle=False, prox_skip=True, loss=loss, it_local=self.it_local, batch_size=self.batch_size) for loss in loss_ids]
        
    def update_trace(self, first_iterations=10):
        super(ProxSkip, self).update_trace()
        self.trace.hs.append(copy.deepcopy(self.h))
        
    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)
