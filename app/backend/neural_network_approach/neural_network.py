import matplotlib.pyplot as plt
import nltk
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
from collections import Counter
from itertools import product
from nltk.tokenize import TweetTokenizer, word_tokenize
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from statistics import mean, mode, median
from torch import optim
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizerFast

nltk.download('punkt')
nltk.download('stopwords')


class Dataset(torch.utils.data.Dataset):
    """
    A class for creating a dataset for training a model
    Attributes:
        texts : iterable - texts used for training
        labels : iterable - lables of the texts used for training
        maxlen : int - maximum length of a sentence for training
        word2token : dict - a dictionary to convert words into digits for training a model
        device : str - device to use for training (gpu or cpu)
    """
    def __init__(self, texts, labels, maxlen, word2token, device):
        self.texts = texts
        self.labels = labels
        self.device = device
        self.maxlen = maxlen
        self.word2token = word2token

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        transformed_text = [self.word2token.get(word, 1) for word in text][:self.maxlen]
        transformed_text = torch.tensor(
            transformed_text + [self.word2token['PAD'] for _ in range(self.maxlen - len(transformed_text))],
            dtype=torch.long, device=self.device)
        attention_ids = torch.tensor(
            [1 for _ in range(len(transformed_text))] + [0 for _ in range(self.maxlen - len(transformed_text))],
            dtype=torch.long, device=self.device)
        return transformed_text, len(transformed_text), attention_ids, label

    def __len__(self):
        return len(self.texts)


class RNNclassifier(nn.Module):
    """
    A class for training an RNN-based classifier
    Attributes:
        device : str - device to use for training (gpu or cpu)
        emb_size : int - size of the word embedding layer
        num_classes : int - number of classes to train on
        dropout : float - ratio of dropout level during the training
        hidden_size : int - size of the hidden layer of the LSTM
    """
    def __init__(self, device, emb_size, num_classes=1, dropout=0.4, hidden_size=100):
        super(RNNclassifier, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.emb_size, self.hidden_size).to(self.device)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=2, bidirectional=True, batch_first=True).to(
            self.device)

        self.linear = nn.Linear(self.hidden_size * 2, self.num_classes).to(self.device)

    def forward(self, tokens, attention_ids, length):
        embs = self.embedding(tokens)
        rnn_out, hidden = self.rnn(embs)
        drop_out = self.dropout(rnn_out)
        output_zero_padding = drop_out.permute([2, 0, 1]) * attention_ids
        output_zero_padding = output_zero_padding.permute([1, 2, 0])
        out = torch.sum(output_zero_padding, 1).T / length
        # out = torch.sum(drop_out, 1).T / length
        out = out.T
        out = self.linear(out)
        return out


def train_model(model, dataloader, dev_dataloader, epoches, optim=optim.RMSprop, lr=0.01):
    """
    A function to train a torch model

    Parameters:
        model : torch model - a model that will be trained
        dataloader : torch dataloader - a dataloader that contains the data for training
        dev_dataloader : torch dataloader - a dataloader that contains the data for validating
        epoches : int - number of epochs during which the training will last
        optim : torch optimizer - an optimizer to use for the network weights update
        lr : float - a learning rate to use for the weights update

    """
    optimizer = optim(model.parameters(), lr=lr)  # Adam, AdamW, Adadelta, Adagrad, SGD, RMSProp
    binary = nn.BCEWithLogitsLoss()
    best_f = 0
    for epoch in range(epoches):
        print(epoch + 1, "epoch")
        t = tqdm(dataloader)
        i = 0
        for sentence, length, attention_ids, label in t:
            pred = model(sentence, attention_ids, length)
            loss = binary(pred.view(-1), label.type(torch.float32))
            if i % 10 == 0:
                torch.save(model.state_dict(), 'model.pt')
                predicted = []
                true = []
                with torch.no_grad():
                    model.eval()
                    for sentence, length, attention_ids, label in dev_dataloader:
                        pred = model(sentence, attention_ids, length)
                        idx = (torch.sigmoid(pred) > 0.5).type(torch.int).item()
                        predicted.append(idx)
                        true.append(label.item())
                    model.train()
                f1 = f1_score(true, predicted, average='macro')
                if f1 > best_f:
                    model.eval()
                    torch.save(model.state_dict(), f"{round(f1, 3)}model.pt")
                    model.train()
                    best_f = f1
                    print("Saving with score", best_f)
            i += 1
            t.set_description(f"loss: {round(float(loss), 3)}, f-macro: {round(f1, 3)}")
            t.refresh()
            loss.backward()
            optimizer.step()
            model.zero_grad()
    return best_f


def evaluate(model, test_dataloader):
    """
    A function to evaluate a torch model

    Parameters:
        model : torch model - a model that will be evaluated
        test_dataloader : torch dataloader - a dataloader that contains the data for evaluating
    """
    predicted = []
    true = []
    with torch.no_grad():
        for sentence, length, attention_ids, label in test_dataloader:
            pred = model(sentence, attention_ids, length)
            idx = (torch.sigmoid(pred) > 0.5).type(torch.int).item()
            predicted.append(idx)
            true.append(label.item())
    print(classification_report(true, predicted))


class NeuralNetwork:
    """
    A class for training and evaluating a neural network model

    Attributes:
        nn_type : str - an NN architecture to use for training (currently only LSTM is supported)
        device : str - device to use for training (gpu or cpu)
    """
    def __init__(self, nn_type, device):
        self.nn_type = nn_type
        self.word2token = {'PAD': 0, 'UNK': 1}
        self.device = device
        self.model = None
        self.tokenize = TweetTokenizer()

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None, ckpt=None, w2t=None):
        if ckpt:
            with open(w2t, 'rb') as handle:
                self.word2token = pickle.load(handle)
            self.model = RNNclassifier(self.device, len(self.word2token), hidden_size=25)
            self.model.load_state_dict(torch.load(ckpt))
        else:
            self.fill_dict(X_train)
            trainds = Dataset(X_train, y_train, 50, self.word2token, self.device)
            devds = Dataset(X_val, y_val, 50, self.word2token, self.device)
            train_dataloader = torch.utils.data.DataLoader(trainds, batch_size=128)
            dev_dataloader = torch.utils.data.DataLoader(devds, batch_size=1)
            self.model = RNNclassifier(self.device, len(self.word2token), 1, 0.4, 25)
            self.model.train()
            train_model(self.model, train_dataloader, dev_dataloader, epoches=20, lr=0.01, optim=optim.Adam)
            with open('word2token.pkl', 'wb') as handle:
                pickle.dump(self.word2token, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.model

    def evaluate(self, X_test, y_test):
        testds = Dataset(X_test, y_test, 50, self.word2token, self.device)
        test_dataloader = torch.utils.data.DataLoader(testds, batch_size=1)
        evaluate(self.model, test_dataloader)

    def predict(self, sent):
        testds = Dataset([self.tokenize.tokenize(sent)], [0], 50, self.word2token, self.device)
        test_dataloader = torch.utils.data.DataLoader(testds, batch_size=1)
        with torch.no_grad():
            for sentence, length, attention_ids, label in test_dataloader:
                pred = self.model(sentence, attention_ids, length)
                idx = (torch.sigmoid(pred) > 0.5).type(torch.int).item()
        return idx, torch.sigmoid(pred).item()

    def fill_dict(self, X_train):
        all_words = set()
        for text in X_train:
            for word in text:
                all_words.add(word)
        for word in all_words:
            self.word2token[word] = len(self.word2token)
