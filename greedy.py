
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class GreedyAlgorithm:
    '''base class for a greedy algorithm'''

    def __init__(self, params_alg):
        self.__dict__.update(params_alg)

    def find_best_element(self):
        self.j = np.argmax(np.abs(np.matmul(self.D, self.f)))
        self.ind.append(self.j)
        self.phi = self.D[self.j]

    def update_remainder(self):
        self.err.append(np.linalg.norm(self.f))

    def approximate(self, f):
        print(f'  calculating {self.__class__.__name__}...')
        self.f = f.copy()
        self.err = [np.linalg.norm(self.f)]
        self.ind, self.proj = [], []
        for i in range(self.max_iter):
            self.find_best_element()
            self.greedy_step()
            self.update_remainder()


class PGA(GreedyAlgorithm):
    '''pure greedy algorithm'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        self.f -= np.dot(self.f, self.phi) * self.phi
        self.proj.append(1)


class OGA(GreedyAlgorithm):
    '''orthogonal greedy algorithm'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        Phi = np.linalg.qr(self.D[np.unique(self.ind)].T)[0].T
        self.f -= np.matmul(np.matmul(Phi, self.f), Phi)
        self.proj.append(len(self.ind))


class NGA(GreedyAlgorithm):
    '''naive greegy algorithm'''

    def __init__(self, params_alg):
        super().__init__(params_alg)

    def greedy_step(self):
        if self.ind.count(self.j) > 1:
            vals, count = np.unique(self.ind, return_counts=True)
            phi_ind = np.where(count > 1)[0]
            Phi = np.linalg.qr(self.D[vals[phi_ind]].T)[0].T
            self.f -= np.matmul(np.matmul(Phi, self.f), Phi)
            self.proj.append(len(phi_ind))
        else:
            self.f -= np.dot(self.f, self.phi) * self.phi
            self.proj.append(1)


class MGA(GreedyAlgorithm):
    '''mixed greedy algorithm'''

    def __init__(self, params_alg, mu):
        super().__init__(params_alg)
        self.mu = mu

    def greedy_step(self):
        self.f -= np.dot(self.f, self.phi) * self.phi
        self.proj.append(1)
        cossim = np.matmul(self.D[self.ind], self.f / np.linalg.norm(self.f))
        mu_ind = np.where(np.abs(cossim) > self.mu)[0]
        if mu_ind.size > 0:
            Phi_ind = np.append(self.j, np.array(self.ind)[mu_ind])
            Phi = np.linalg.qr(self.D[np.unique(Phi_ind)].T)[0].T
            self.f -= np.matmul(np.matmul(Phi, self.f), Phi)
            self.proj[-1] = len(Phi_ind)


class SetupExperiments:
    '''setup and run the experiments'''

    def __init__(self, params_exp):
        self.__dict__.update(params_exp)
        np.random.seed(self.seed)

    def generate_dictionary(self):
        if self.type_d == 'gauss':
            self.D = np.random.randn(self.num_d, self.dim)
        elif self.type_d == 'uniform':
            self.D = 2 * np.random.rand(self.num_d, self.dim) - 1
        self.D /= np.linalg.norm(self.D, axis=1, keepdims=True)

    def generate_element(self):
        f_ind = np.random.randint(self.num_d, size=self.f_card)
        f_coef = np.random.randn(self.f_card)
        self.f = np.matmul(f_coef, self.D[f_ind]) + self.f_noise * np.random.rand(self.dim)
        self.f /= np.linalg.norm(self.f)

    def run(self):
        self.err = {'pga': {}, 'oga': {}, 'nga': {}, 'mga': {}}
        self.ind = {'pga': {}, 'oga': {}, 'nga': {}, 'mga': {}}
        self.proj = {'pga': {}, 'oga': {}, 'nga': {}, 'mga': {}}
        for t in range(self.num_tests):
            print(f'running test {t+1}/{self.num_tests}:')
            self.generate_dictionary()
            self.generate_element()
            self.params_alg = {'D': self.D, 'max_iter': self.max_iter}
            # Pure Greedy Algorithm
            self.pga = PGA(self.params_alg)
            self.pga.approximate(self.f)
            self.err['pga'].update({t: self.pga.err})
            self.ind['pga'].update({t: self.pga.ind})
            self.proj['pga'].update({t: np.cumsum(self.pga.proj)})
            # Orthogonal Greedy Algorithm
            self.oga = OGA(self.params_alg)
            self.oga.approximate(self.f)
            self.err['oga'].update({t: self.oga.err})
            self.ind['oga'].update({t: self.oga.ind})
            self.proj['oga'].update({t: np.cumsum(self.oga.proj)})
            # Naive Greedy Algorithm
            self.nga = NGA(self.params_alg)
            self.nga.approximate(self.f)
            self.err['nga'].update({t: self.nga.err})
            self.ind['nga'].update({t: self.nga.ind})
            self.proj['nga'].update({t: np.cumsum(self.nga.proj)})
            # Mixed Greedy Algorithm
            self.mga = MGA(self.params_alg, mu=.5*self.dim/self.num_d)
            self.mga.approximate(self.f)
            self.err['mga'].update({t: self.mga.err})
            self.ind['mga'].update({t: self.mga.ind})
            self.proj['mga'].update({t: np.cumsum(self.mga.proj)})

    def plot_err(self):
        pga_err = pd.DataFrame(self.err['pga'])
        oga_err = pd.DataFrame(self.err['oga'])
        nga_err = pd.DataFrame(self.err['nga'])
        mga_err = pd.DataFrame(self.err['mga'])
        # compute stats
        pga_mean, pga_std = pga_err.mean(axis=1), pga_err.std(axis=1)
        oga_mean, oga_std = oga_err.mean(axis=1), oga_err.std(axis=1)
        nga_mean, nga_std = nga_err.mean(axis=1), nga_err.std(axis=1)
        mga_mean, mga_std = mga_err.mean(axis=1), mga_err.std(axis=1)
        # plot errors
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(np.arange(len(pga_err)), pga_mean, linewidth=3, label='PGA')
        ax.plot(np.arange(len(oga_err)), oga_mean, linewidth=3, label='OGA')
        ax.plot(np.arange(len(nga_err)), nga_mean, linewidth=3, label='NGA')
        ax.plot(np.arange(len(mga_err)), mga_mean, linewidth=3, label='MGA')
        ax.fill_between(np.arange(len(pga_err)), pga_mean - pga_std, pga_mean + pga_std, alpha=.5)
        ax.fill_between(np.arange(len(oga_err)), oga_mean - oga_std, oga_mean + oga_std, alpha=.5)
        ax.fill_between(np.arange(len(nga_err)), nga_mean - nga_std, nga_mean + nga_std, alpha=.5)
        ax.fill_between(np.arange(len(mga_err)), mga_mean - mga_std, mga_mean + mga_std, alpha=.5)
        ax.set_xlabel('algorithm iterations')
        ax.set_ylabel('approximation error')
        ax.set_yscale('log')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_coef(self):
        pga_coef = pd.DataFrame(self.proj['pga'])
        oga_coef = pd.DataFrame(self.proj['oga'])
        nga_coef = pd.DataFrame(self.proj['nga'])
        mga_coef = pd.DataFrame(self.proj['mga'])
        # compute stats
        pga_mean, pga_std = pga_coef.mean(axis=1), pga_coef.std(axis=1)
        oga_mean, oga_std = oga_coef.mean(axis=1), oga_coef.std(axis=1)
        nga_mean, nga_std = nga_coef.mean(axis=1), nga_coef.std(axis=1)
        mga_mean, mga_std = mga_coef.mean(axis=1), mga_coef.std(axis=1)
        # plot coefficient change count
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(np.arange(len(pga_coef)), pga_mean, linewidth=3, label='PGA')
        ax.plot(np.arange(len(oga_coef)), oga_mean, linewidth=3, label='OGA')
        ax.plot(np.arange(len(nga_coef)), nga_mean, linewidth=3, label='NGA')
        ax.plot(np.arange(len(mga_coef)), mga_mean, linewidth=3, label='MGA')
        ax.fill_between(np.arange(len(pga_coef)), pga_mean - pga_std, pga_mean + pga_std, alpha=.5)
        ax.fill_between(np.arange(len(oga_coef)), oga_mean - oga_std, oga_mean + oga_std, alpha=.5)
        ax.fill_between(np.arange(len(nga_coef)), nga_mean - nga_std, nga_mean + nga_std, alpha=.5)
        ax.fill_between(np.arange(len(mga_coef)), mga_mean - mga_std, mga_mean + mga_std, alpha=.5)
        ax.set_xlabel('algorithm iterations')
        ax.set_ylabel('cumulative coefficient changes')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    params_exp = {'dim': 1000, 'num_d': 5000, 'type_d': 'gauss', 'f_card': 200, 'f_noise': .01,
                  'max_iter': 500, 'num_tests': 1, 'seed': 0}
    exp = SetupExperiments(params_exp)
    exp.run()
    exp.plot_err()
    exp.plot_coef()

