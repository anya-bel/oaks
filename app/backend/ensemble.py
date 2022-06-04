import pickle
from time import time

import torch

from .cross_checking_approach.cross_checking_module import CheckText
from .feature_based_approach.feature_based import FeatureBased
from .neural_network_approach.neural_network import NeuralNetwork, RNNclassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ensembling:
    def __init__(self, name='voting', device=device, nn_ckpt='../../backend/models/newmodel.pt',
                 nn_w2t='../../backend/models/word2token.pkl', logreg_ckpt='../../backend/models/lr_ensemble.sav'):
        self.name = name
        self.device = device
        self.feba = FeatureBased()
        self.crch = CheckText()
        self.nene = NeuralNetwork('rnn', self.device)
        self.nene.fit(ckpt=nn_ckpt, w2t=nn_w2t)
        if self.name not in ['voting', 'logreg']:
            raise ValueError()
        if self.name == 'logreg':
            self.logreg = pickle.load(open(logreg_ckpt, 'rb'))

    def predict(self, text):
        st = time()
        pred1 = True if not self.feba.predict(text, params={'load': -0.3, 'rep': 2, 'plur': 2, 'spell': 2, 'punct': 0,
                                                            'excess': 0, 'past': 0.0})[0] else False
        f = time()
        print(f'Feature-based predicton made in {round(f - st, 2)}s')
        fullpred2 = self.crch.check(text)
        pred2 = fullpred2[0]
        c = time()
        print(f'Cross-cheking predicton made in {round(c - f, 2)}s')
        fullpred3 = self.nene.predict(text)
        pred3 = True if bool(fullpred3[0]) else False
        n = time()
        print(f'NN predicton made in {round(n - c, 2)}s')
        print(pred1, pred2, pred3)
        preds = [pred1, pred2, pred3]
        if self.name == 'voting':
            if preds.count(True) > preds.count(False):
                return True, preds
            return False, preds
        else:
            preds = [[pred1, fullpred2[1], fullpred3[1]]]
            pred = self.logreg.predict(preds)
            if pred == 'true':
                return True, self.logreg.predict_proba(preds)[0][0]
            else:
                return False, self.logreg.predict_proba(preds)[0][0]
