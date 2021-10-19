from math import exp, log, sqrt
from random import random


class FTRL_proximal:
    """
        Origin paper:
            https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf

        Based on code from:
            https://www.kaggle.com/jiweiliu/ftrl-starter-code
            https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
    """
    def __init__(self, alpha=0.005, beta=1,
                 lambda_1=0, lambda_2=1,
                 feature_vec_d=100000, epoch=1):
        # parameters related to adaptive learning rate
        self.alpha = alpha
        self.beta = beta
        # L1 and L2 coefficience
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        # parameters about feature interaction
        # a feature vector of length equals to (n_unique_values_per_category * n_categories)  is good
        self.feature_vec_d = feature_vec_d
        self.epoch = epoch
        # n, z corresponding to the denote of origin paper
        self.z = [0] * self.feature_vec_d  # all weights for LR model
        self.n = [0] * self.feature_vec_d  # every weight's squared sum of previous gradient

    @staticmethod
    def logloss_binary(y_p, y):
        """
            Compute binary logloss of the predict result

            Parameters
            ----------
            y_p : [float]
                [predicted value]
            y : [float or int]
                [real value]
            
            Returns
            -------
            [float]
                [logloss value]
        """
        # keep p in bound, avoid log(0)
        y_p = max(min(y_p, 1 - 10e-15), 10e-15)
        return -log(y_p) if y == 1 else -log(1-y_p)

    def get_none_zero_entry_indices(self, inputs):
        """
            Use hash trick to get the none zero value indices in the one-hot encoded feature vector of the sample
            It treats all input value as category feature, numeric values can be convert to category using range bucket

        Parameters
        ----------
        inputs : [dict]
            [{feature_name: feature_value} of one sample]
        
        Returns
        -------
        [list]
            [list of the hot indices in the feature vector for this sample]
        """
        X = [0]  # refer to the bias term index in the weights
        for key, value in inputs.items():
            # use hash trick to provide unique value-feature_name pair to generate unique value for each feature value
            # one difference from standard one-hot is that indices for values of one category may not be continous,
            # wich is totally fine for the model.
            one_hot_index = abs(hash(str(value) + "-" + key)) % self.feature_vec_d
            X.append(one_hot_index)
        return X

    @staticmethod
    def sgn(x):
        """
            Function to get the sign of the number
        """
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    @staticmethod
    def sigmoid(x):
        # bounded for eliminate extrem values
        x = max(min(x, 35), -35)
        return 1 / (1+exp(-x))

    def predict(self, inputs):
        """
            Implementation of predict process shown in the paper's algorithm table: 
                Algorithm 1 --> line 5 to line 9 (Predict ..... computed above)
        
        Parameters
        ----------
        inputs : [dict]
            [{feature_name: feature_value} of one sample]
        
        Returns
        -------
        [tuple]
            [wTx: predict value]
            [I: none zero indices of in feature vector]
            [w: temporal weights for this round]
        """
        # predict value (dot product of w and x)
        wTx = 0
        # get I = {i | x[i] != 0}
        I = self.get_none_zero_entry_indices(inputs)
        # compute independent new w per sample
        # w = [None for _ in range(self.feature_vec_d)]
        # feature vec is very sparse, which means `I` contains few indices in full feature vec.
        # use dict here is much better than list, since we only need to store i in `I`
        # a list with full length of self.feature_vec_d will waste tons of memory and time
        w = {}
        for i in I:
            z_i = self.z[i]
            if abs(z_i) <= self.lambda_1:
                w[i] = 0
            else:
                sgn = -1 if z_i < 0 else 1
                w[i] = (self.sgn(z_i)*self.lambda_1-z_i) / ((self.beta+sqrt(self.n[i]))/self.alpha + self.lambda_2)
            wTx += w[i] * 1  # none zero x in the feature vector must be 1
        return self.sigmoid(wTx), I, w

    def update_weights(self, I, w, p, y):
        """
            Implementation of update process shown in the paper's algorithm table: 
                Algorithm 1 --> line 10 to line 15 ()
            
        """
        for i in I:
            g = (p-y) * 1  # xi is 1 for all i
            n = self.n[i]
            # sigma is the adaptive learning rate, which vary from sample to
            # sample and from weight to weight
            sigma = (sqrt(n+g**2)-sqrt(n))/self.alpha
            self.z[i] += (g - sigma*w[i])
            self.n[i] += g**2


