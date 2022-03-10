
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from greedy_algorithms import *

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class SetupExperiments:
    '''setup and run the experiments'''

    def __init__(self, params_exp):
        self.__dict__.update(params_exp)
        self.module = __import__('__main__')
        os.makedirs('./images/', exist_ok=True)
        np.random.seed(self.seed)

    def generate_dictionary(self):
        '''generate random dictionary'''
        if self.type_d == 'gauss':
            self.D = np.random.randn(self.num_d, self.dim)
        elif self.type_d == 'uniform':
            self.D = 2 * np.random.rand(self.num_d, self.dim) - 1
        elif self.type_d == 'cluster':
            self.D = np.random.randn(self.dim) + 2*np.random.randn(self.num_d, self.dim)
        self.D /= np.linalg.norm(self.D, axis=1, keepdims=True)

    def generate_element(self):
        '''generate target element'''
        if self.type_d == 'cluster':
            self.f = np.random.randn(self.dim)
            self.f = self.f - np.dot(self.f, self.D[0]) * self.D[0]
        else:
            f_ind = np.random.randint(self.num_d, size=self.f_card)
            f_coef = np.random.randn(self.f_card)
            self.f = np.matmul(f_coef, self.D[f_ind]) + self.f_noise * np.random.rand(self.dim)
        self.f /= np.linalg.norm(self.f)

    def run(self):
        '''iteratively run experiments'''
        self.err = {alg: {} for alg in self.algos}
        self.ind = {alg: {} for alg in self.algos}
        self.coef = {alg: {} for alg in self.algos}
        for t in range(self.num_tests):
            print(f'running test {t+1}/{self.num_tests}:')
            self.generate_dictionary()
            self.generate_element()
            self.params_alg = {'D': self.D, 'max_iter': self.max_iter, 'mu': self.mu}
            for alg in self.algos:
                alg_class = getattr(self.module, alg)(self.params_alg)
                alg_class.approximate(self.f)
                self.err[alg].update({t: alg_class.err})
                self.ind[alg].update({t: alg_class.ind})
                self.coef[alg].update({t: np.cumsum(alg_class.coef)})

    def plot_err(self):
        '''plot approximation errors'''
        fig, ax = plt.subplots(figsize=(8,5))
        for alg in self.algos:
            alg_err = pd.DataFrame(self.err[alg])
            alg_mean, alg_std = alg_err.mean(axis=1), alg_err.std(axis=1)
            alg_itr = np.arange(len(alg_err))
            ax.plot(alg_itr, alg_mean, linewidth=3, label=alg)
            ax.fill_between(alg_itr, alg_mean-alg_std, alg_mean+alg_std, alpha=.5)
        ax.set_yscale('log')
        ax.set_xlabel('algorithm iterations')
        ax.set_ylabel('approximation error')
        ax.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('./images/errors.png', dpi=300, format='png')
        plt.show()

    def plot_coef(self):
        '''plot number of coefficient changes'''
        fig, ax = plt.subplots(figsize=(8,5))
        for alg in self.algos:
            alg_coef = pd.DataFrame(self.coef[alg])
            alg_mean, alg_std = alg_coef.mean(axis=1), alg_coef.std(axis=1)
            alg_itr = np.arange(len(alg_coef))
            ax.plot(alg_itr, alg_mean, linewidth=3, label=alg)
            ax.fill_between(alg_itr, alg_mean-alg_std, alg_mean+alg_std, alpha=.5)
        ax.set_yscale('log')
        ax.set_xlabel('algorithm iterations')
        ax.set_ylabel('cumulative coefficient changes')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('./images/coeffs.png', dpi=300, format='png')
        plt.show()


if __name__ == '__main__':
    '''setup and run experiments'''
    params_exp = {'dim': 1000, 'num_d': 5000, 'type_d': 'cluster', 'f_card': 250, 'f_noise': .01,
                  'max_iter': 500, 'num_tests': 1, 'seed': 0, 'mu': .5,
                  'algos': ['PGA', 'OGA', 'MGA_g_prod', 'MGA_prod']}
    exp = SetupExperiments(params_exp)
    exp.run()
    exp.plot_err()
    exp.plot_coef()

