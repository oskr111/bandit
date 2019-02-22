# 参考
# https://hagino3000.blogspot.com/2014/05/banditalgo4.html

exec(open('core.py', encoding='utf-8').read())

import random
import matplotlib.pylab as plt
import numpy as np

def plot_results(simulate_num, horizon, best_arm, results):
    fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(11, 5))
    x = range(horizon)
    plot1, plot2, plot3 = axes1[0], axes1[1], axes1[2]

    for result in test_results:
        accuracy = np.zeros(horizon)
        reward_ave = np.zeros(horizon)
        cumulative_rewards = np.zeros(horizon)

        epsilon = result[0]
        chosen_arms_mat = result[1]
        rewards_mat = result[2]
        cumulative_rewards_mat = result[3]

        for i in range(horizon):
            best_arm_selected_count = len(list(filter(lambda choice: choice == best_arm, chosen_arms_mat[:,i])))
            accuracy[i] = best_arm_selected_count / float(simulate_num)
            reward_ave[i] = np.average(rewards_mat[:,i])
            cumulative_rewards[i] = np.average(cumulative_rewards_mat[:,i])

        plot1.plot(x, accuracy, label='%10.2f' % epsilon)
        plot2.plot(x, reward_ave, label='%10.2f' % epsilon)
        plot3.plot(x, cumulative_rewards, label='%10.2f' % epsilon)

    plot1.legend(loc=4)
    plot1.set_xlabel('Time')
    plot1.set_ylabel('Probability of Selecting Best Arm')
    plot1.set_title('Accuracy of the \nEpsilon Greedy Algorithm')

    plot2.legend(loc=4)
    plot2.set_xlabel('Time')
    plot2.set_ylabel('Average Reward')
    plot2.set_title('Performance of the \nEpsilon Greedy Algorithm')

    plot3.legend(loc=4)
    plot3.set_xlabel('Time')
    plot3.set_ylabel('Cumulative Reward of Chosen Arm')
    plot3.set_title('Cumulative Reward of the \nEpsilon Greedy Algorithm')

    plt.show()

SIMULATE_NUM = 5000
HORIZON = 250

# main
means = [0.1, 0.1, 0.1, 0.1, 0.9]
print(means)
n_arms = len(means)
random.shuffle(means)
arms = list(map(lambda mu: BernoulliArm(mu), means))
best_arm = ind_max(means)
print("Best arm is" + str(best_arm))

"""
# 腕200本
means = np.random.rand(20)
print(means)
n_arms = len(means)
random.shuffle(means)
arms = list(map(lambda mu: BernoulliArm(mu), means))
best_arm = np.array(means).argmax()
print("Best arm is" + str(best_arm))
"""

test_results = []
for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
#for epsilon in [1.0]:
    algo = EpsilonGreedy(epsilon, [], [])
    algo.initialize(n_arms)
    chosen_arms, rewards, cumulative_rewards = test_algorithm(algo, arms, SIMULATE_NUM, HORIZON)
    test_results.append([epsilon, chosen_arms, rewards, cumulative_rewards])
plot_results(SIMULATE_NUM, HORIZON, best_arm, test_results)
