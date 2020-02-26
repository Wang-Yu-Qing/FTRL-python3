import pickle
import pandas as pd
from csv import DictReader
from FTRL import FTRL_proximal


def train():
    # Train the model using streaming data with epoch as 1, every sample only used once 
    # in a real off-line training senario, we can set training epoch larger than 1
    model = FTRL_proximal()
    with open("data/train.csv", "r") as f:
        row_generator = DictReader(f)
        # the loss is overall average loss on every sample, so 
        # it will be larger than the loss of test set because of
        # the poor predictions at the training begining.
        loss = 0
        losses = []
        for t, row in enumerate(row_generator):
            y = int(row["target"])
            del row["target"]
            y_p, I, w = model.predict(row)
            loss += FTRL_proximal.logloss_binary(y_p, y)
            model.update_weights(I, w, y_p, y)
            if t % 1000 == 0:
                print('[Training] {} samples processed, current loss: {}'.format(t+1,
                                                                    round(loss/(t+1), 5)))
                losses.append((t+1, loss/(t+1)))
    print("Training done.")
    with open("model.pickle", "wb") as f:
        f.write(pickle.dumps(model))

def evaluate():
    """
    The test data set doesn't have true label. To evaluate and update the model,
    one should do it on kaggle website and submit a submit file

    The following code simulate the real online predicting and learning senario, 
    if true label is provided, the model can be updated with the streaming data
    """
    with open("model.pickle", "rb") as f:
        model = pickle.loads(f.read())
    with open("data/test.csv", "r") as f:
        row_generator = DictReader(f)
        loss = 0
        losses = []
        print("Start testing.")
        for t, row in enumerate(row_generator):
            y = int(row["target"])  # actually, no target in the test data
            del row["target"]
            y_p, I, w = model.predict(row)
            loss += FTRL_proximal.logloss_binary(y_p, y)
            model.update_weights(I, w, y_p, y)
            if t % 1000 == 0:
                print('[Evaluating] {} samples processed, current loss: {}'.format(t+1,
                                                                    round(loss/(t+1), 5)))
                losses.append((t+1, loss/(t+1)))


def make_submission_file():
    """
    make kaggle submission file
    """
    with open("model.pickle", "rb") as f:
        model = pickle.loads(f.read())
    with open("data/test.csv", "r") as f:
        row_generator = DictReader(f)
        loss = 0
        losses = []
        print("Start testing.")
        with open("submit.csv", "w") as f:
            f.write("ID,target\n")
            for t, row in enumerate(row_generator):
                y_p, I, w = model.predict(row)
                f.write("{},{}\n".format(row["ID"], y_p))
            


if __name__ == "__main__":
    train()
    # evaluate()
    make_submission_file()