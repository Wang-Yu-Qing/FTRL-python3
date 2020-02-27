# Logistic Regression using FTRL for CTR prediction

An implementation of logistic regression using google's FTRL-proximal online predicting and adaptive learning method with python3 to solve a CTR prediction problem with features in a very high dimension space.

**Origin paper:** 

https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf 

**The code is based on the following codes from kaggle:**

https://www.kaggle.com/jiweiliu/ftrl-starter-code

https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory

### dataset
Dataset used: https://www.kaggle.com/c/springleaf-marketing-response/data

### some pros of FTRL
1. It is an online learning method, samples are fed in stream and the model is updated in real-time.
2. The learning rate is adaptive per-coordinate, which means the learning rates of all weights are varying adaptively.
3. Using the strong L1 and L2 regularization, the model is going to be sparse, saving lots of memory.
4. A good balance between model performance (e.g. AUC) and model sparsity.