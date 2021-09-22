
import numpy as np


class GreedyAlgorithm:
    '''base class for a greedy algorithm'''

    def __init__(self, params_alg):
        self.__dict__.update(params_alg)

    def approximate(self, f):
        print(f'  calculating {self.__class__.__name__}...')
        self.f = f.copy()
        self.err = [np.linalg.norm(self.f)]
        self.ind, self.coef = [], []
        for i in range(self.max_iter):
            self.find_best_element()
            self.greedy_step()
            self.err.append(np.linalg.norm(self.f))

    def find_best_element(self):
        self.j = np.argmax(np.abs(np.matmul(self.D, self.f)))
        self.ind.append(self.j)
        self.phi = self.D[self.j]


class PGA(GreedyAlgorithm):
    '''pure greedy algorithm'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        self.f -= np.dot(self.f, self.phi) * self.phi
        self.coef.append(1)


class OGA(GreedyAlgorithm):
    '''orthogonal greedy algorithm'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        Phi_ind = np.unique(self.ind)
        Phi_ort = np.linalg.qr(self.D[Phi_ind].T)[0].T
        self.f -= np.matmul(np.matmul(Phi_ort, self.f), Phi_ort)
        self.coef.append(len(Phi_ind))


class MGA_g_cos(GreedyAlgorithm):
    '''mixed greedy algorithm (cossim)'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        self.f -= np.dot(self.f, self.phi) * self.phi
        self.coef.append(1)
        cossim = np.matmul(self.D[self.ind], self.f / np.linalg.norm(self.f))
        mu_ind = np.where(np.abs(cossim) > self.mu)[0]
        if mu_ind.size > 0:
            Phi_ind = np.unique(np.append(self.j, np.array(self.ind)[mu_ind]))
            Phi_ort = np.linalg.qr(self.D[Phi_ind].T)[0].T
            self.f -= np.matmul(np.matmul(Phi_ort, self.f), Phi_ort)
            self.coef[-1] = len(Phi_ind)


class MGA_g_prod(GreedyAlgorithm):
    '''mixed greedy algorithm (proj+prod)'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        prod = np.abs(np.dot(self.f, self.phi))
        self.f -= np.dot(self.f, self.phi) * self.phi
        self.coef.append(1)
        mu_ind = np.where(np.abs(np.matmul(self.D[self.ind], self.f)) > self.mu * prod)[0]
        if mu_ind.size > 0:
            Phi_ind = np.unique(np.append(self.j, np.array(self.ind)[mu_ind]))
            Phi_ort = np.linalg.qr(self.D[Phi_ind].T)[0].T
            self.f -= np.matmul(np.matmul(Phi_ort, self.f), Phi_ort)
            self.coef[-1] = len(Phi_ind)


class MGA_prod(GreedyAlgorithm):
    '''mixed greedy algorithm (prod)'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        prod = np.abs(np.dot(self.f, self.phi))
        mu_ind = np.where(np.abs(np.matmul(self.D[self.ind], self.f)) > self.mu * prod)[0]
        Phi_ind = np.unique(np.append(self.j, np.array(self.ind)[mu_ind]))
        Phi_ort = np.linalg.qr(self.D[Phi_ind].T)[0].T
        self.f -= np.matmul(np.matmul(Phi_ort, self.f), Phi_ort)
        self.coef.append(len(Phi_ind))


class MGA_mu(GreedyAlgorithm):
    '''mixed greedy algorithm (adaptive mu)'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        prod = np.abs(np.dot(self.f, self.phi))
        self.f -= np.dot(self.f, self.phi) * self.phi
        try:
            mu = np.sqrt(1 - np.dot(self.D[self.ind[-1]], self.D[self.ind[-2]])**2)
        except:
            mu = 1
        mu_ind = np.where(np.abs(np.matmul(self.D[self.ind], self.f)) >= mu * np.linalg.norm(self.f))[0]
        Phi_ind = np.unique(np.append(self.j, np.array(self.ind)[mu_ind]))
        Phi_ort = np.linalg.qr(self.D[Phi_ind].T)[0].T
        self.f -= np.matmul(np.matmul(Phi_ort, self.f), Phi_ort)
        self.coef.append(len(Phi_ind))

