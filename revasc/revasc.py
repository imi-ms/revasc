import os

from fastai.learner import load_learner
from tsai.inference import get_X_preds
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Revasc(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, epochs=100, cpu=False,
                 rule_in_threshold=.05433071, rule_out_threshold=.00671125):
        self.model = None
        self.classes_ = np.array([0, 1])
        self.random_state = random_state
        self.epochs = epochs
        self.cpu = cpu
        self.rule_in_threshold = rule_in_threshold,
        self.rule_out_threshold = rule_out_threshold

        self.models = []

    def predict(self, X, bs=64):
        preds_int = [model.get_X_preds(X, bs=bs)[2].astype(int).astype(int) for model in self.models]
        return np.median(preds_int, axis=0)

    def predict_proba(self, X, bs=64):
        preds = [model.get_X_preds(X, bs=bs)[0].numpy()[:, :] for model in self.models]
        return np.mean(preds, axis=0)

    def predict_risk(self, X, bs=64):
        prob = self.predict_proba(X, bs=bs)
        risk = []
        for i in range(len(X)):
            if prob[i, 1] >= self.rule_in_threshold:
                risk.append("high risk")
            elif prob[i, 1] >= self.rule_out_threshold:
                risk.append("intermediate risk")
            else:
                risk.append("low risk")
        return np.array(risk)

    def load(self, model_path):
        self.models = []
        for i in range(10):
            self.models.append(load_learner(os.path.join(model_path, str(i)+".pkl"), cpu=self.cpu))
