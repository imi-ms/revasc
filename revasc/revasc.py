import pandas as pd
import os
import numpy as np
import torch

import os

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, GroupShuffleSplit
from tsai.all import *
from fastai.metrics import RocAucBinary

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import fastai.learner as fastail
from fastai.callback.tracker import ReduceLROnPlateau, EarlyStoppingCallback
from fastai.callback.training import GradientAccumulation, ShortEpochCallback
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from fastai.metrics import BalancedAccuracy, RocAucBinary


class Revasc(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, epochs=100, cpu=False,
                 rule_in_threshold=.0271980346, rule_out_threshold=.0043238043):
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
        prob = self.predict_proba(X, bs=64)
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
