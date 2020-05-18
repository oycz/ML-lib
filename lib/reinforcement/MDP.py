import numpy as np


class MDP:
    # n: number of states
    # R: Reward function
    # P: transition matrix
    def __init__(self, n, R, gamma,
                 max_iteration=float('inf'), threshold=1e-6):
        self.n = n
        self.R = R
        self.V, self.policy = None, None
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.iteration = 0
        self.fit()

    def fit(self):
        V = [np.random.random(self.n)*10]
        policy = []
        while True:
            new_V = np.zeros(n)
            new_policy = np.zeros(n)
            if self.iteration >= self.max_iteration:
                break
            for _ in range(5*self.n):
            # for i in range(self.n):
                i = np.random.randint(0, 4)
                t = np.array(self.R[i])
                t += self.gamma * V[-1]
                new_V[i] = np.max(t)
                new_policy[i] = np.argmax(t)+1
            V += [new_V]
            policy += [new_policy]
            self.iteration += 1
            if np.sum(V[-1] == V[-2]) == self.n:
                # convergence
                break
        self.V, self.policy = V, policy


if __name__ == "__main__":
    for i in range(100):
        n = 4
        R = np.array([[0  , 1  , 0.5, 0.5],
                      [0.5, 0  , 1  , 0.5],
                      [-1 , 0.5, 0  , 0.5],
                      [-1 , 0.5, 0.5, 0  ]])
        m1 = MDP(4, R, .8)
        m2 = MDP(4, R, .01)
        print(m1.policy[-1])
        print(m2.policy[-1])