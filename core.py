import numpy as np
import math
import random

class UCB1():
    def __init__(self, temperature, counts, values):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            # すべての腕を少なくとも一回は引く
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            # total_counts > counts[arm]ならbounsは大きくなる
            # =未知のものを見つけ出そうとする
            # 明示的に好奇心の強いアルゴリズム
            # 好奇心ボーナス
            bouns = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bouns
        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return

def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1

class Softmax:
    def __init__(self, temperature, counts, values):
        # self.temperature = temperature
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0 for col in range(n_arms)]
        return

    def select_arm(self):
        t = sum(self.counts) + 1
        temperature = 1 / math.log(t + 0.0000001)

        z = sum([math.exp(v / temperature) for v in self.values])
        probs = [math.exp(v / temperature) / z for v in self.values]
        return categorical_draw(probs)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return

class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        t = sum(self.counts) + 1
        # アニール
        delta = 0.0000001
        epsilon = self.epsilon / (abs(math.log(math.log(t + delta)+1+delta)))

        if random.random() > epsilon:
            return ind_max(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

def ind_max(x):
    m = max(x)
    return x.index(m)

def test_algorithm(algo, arms, num_sims, horizon):
    """
    algo:テストしたいバンディットアルゴリズム
    arms:プレイをシミュレートしたい腕の配列
    sim:各シミュレートで生じるノイズを相殺し、平均値を出すために
    必要なシミュレーションの設定回数
    horizon:各シミュレーションにおいて、それぞれのアルゴリズムが腕を引いても良い回数
    """
    # 試行毎に選んだ腕
    chosen_arms = np.zeros((num_sims, horizon), dtype=int)
    # 試行毎に得られた報酬
    rewards = np.zeros((num_sims, horizon))
    # 累積報酬
    cumulative_rewards = np.zeros((num_sims, horizon))

    #sim_nums = [0.0 for i in range(num_sims * horizon)]
    #times = [0.0 for i in range(num_sims * horizon)]

    for sim in range(num_sims):
        algo.initialize(int(len(arms)))
        for t in range(horizon):
            chosen_arm = algo.select_arm()
            chosen_arms[sim, t] = chosen_arm

            #sim_nums[index] = sim
            #times[index] = t

            reward = arms[chosen_arm].draw()
            rewards[sim, t] = reward

            algo.update(chosen_arm, reward)
        cumulative_rewards[sim] = rewards[sim].cumsum()

    return chosen_arms, rewards, cumulative_rewards
