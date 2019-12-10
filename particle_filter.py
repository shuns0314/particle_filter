
import datetime

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt


class ParticalFilter:
    def __init__(self, observed, n_particles=100, num=10):
        # try
        self.num = num
        # 粒子数
        self.n_particles = n_particles
        # 観測値
        self.observed = observed
        # 状態xの初期値として、観測値の平均と標準偏差から算出する。
        mean = np.mean(self.observed)
        alpha = np.std(self.observed)
        self.state = mean + alpha * randn(self.n_particles, len(self.observed))
        self.last_state = self.state
        self.alpha = 0
        self.beta = 0
        self.y_t = np.zeros(n_particles)

    def gaussian_likelihood(self, y_t, x_t, sigma):
        """ガウシアン分布の尤度関数.

        x_tが定まった時、y_tが定まる.
        """
        return np.exp(- (y_t - x_t)**2 / (2 * sigma)) / (2 * np.pi * sigma)**0.5

    def resample(self, state, likelihood):
        """累積和のルーレット的なやつ."""
        # 正規化
        s_likelihood = likelihood / sum(likelihood)
        # 累積和
        cum_likelihood = np.cumsum(s_likelihood)
        # リサンプルの初期位置
        start = np.random.random() / self.n_particles
        # resample
        particles = np.zeros(self.n_particles)
        particle_index = 0
        for index in range(self.n_particles):
            while start < cum_likelihood[index]:
                particles[particle_index] = state[index]
                start += 1 / self.n_particles
                particle_index += 1
        return particles

    def calc(self):
        total_likelihood = 0
        for _ in range(self.num):
            alpha = 10 ** (np.random.randint(-1, 3))
            beta = np.std(self.observed) * np.random.random() * 2
            # print(f"alpha: {alpha}")
            # print(f"beta: {beta}")
            for time in range(1, len(self.observed)):
                # 予測
                # print(f"状態 t={time}: {self.state[:, time-1][0:5]}")
                self.state[:, time] = self.state[:, time-1] + alpha * randn(self.n_particles)
                # print(f"予測 t={time}: {self.state[:, time][0:5]}")
                # print(f"観測値: {self.observed[time]}")

                # filtering(尤度計算＋リサンプル)
                w_t = self.gaussian_likelihood(
                    y_t=self.observed[time],
                    x_t=self.state[:, time],
                    sigma=beta)
                # print(w_t)
                self.state[:, time] = self.resample(self.state[:, time], w_t)

            likelihood = np.sum(np.log(np.sum(self.state, axis=0)/self.n_particles))
            # print(likelihood)
            if total_likelihood < likelihood:
                total_likelihood = likelihood
                self.alpha = alpha
                self.beta = beta
                self.last_state = self.state
                plt.plot(np.mean(self.state, axis=0))

        return self.alpha, self.beta, self.last_state
