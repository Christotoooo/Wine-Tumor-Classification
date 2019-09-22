import numpy as np
import math
class LogisticRegression:
    def __init__(self, X_features, Y_quality):
        # split the training data
        self.X_features = X_features
        self.Y_quality = Y_quality
        self.weight = np.full((self.X_features.shape[1], 1), 0)

    def fit(self, learning_rate, gradient_descent_iterations):
        for it in range(gradient_descent_iterations):
            weight_old = self.weight
            for i in range(self.X_features.shape[0]):
                sigma = np.matmul(np.transpose(weight_old), np.transpose(self.X_features[i]))
                self.weight = np.add(
                    self.weight,
                    (learning_rate * (self.Y_quality[i] - self.logisitic(sigma[0])) * self.X_features[i]).reshape(self.X_features.shape[1], 1))
            # print(it, self.weight)

    def predict(self, input):
        target_evaluation = np.full((input.shape[0], 1), 0)
        for i in range(input.shape[0]):
            log_odds_ratio = np.matmul(np.transpose(self.weight), input[i].reshape(input[i].shape[0], 1))
            target_evaluation[i] = 0 if self.logisitic(log_odds_ratio) < 0.5 else 1
        return target_evaluation

    def logisitic(self, log_odds_ratio):
        if log_odds_ratio < 0:
            return 1 - 1 / (1 + math.exp(log_odds_ratio))
        else:
            return 1 / (1 + math.exp(-log_odds_ratio))
