"""
References:
1. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. International Conference on Learning Representations. https://openreview.net/forum?id=8qDwejCuCN
2. https://openreview.net/forum?id=8qDwejCuCN

Acknowledgements:
- https://github.com/sanatonek/TNC_representation_learning?tab=readme-ov-file
- https://seunghan96.github.io/cl/ts/(CL_code3)TNC/
"""

import os
import torch
import numpy as np
import pickle
import random
import argparse
import matplotlib.pyplot as plt

from tnc.models import RnnEncoder, StateClassifier, E2EStateClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_run_encoder(model, train_loader, test_loader, flatten=False, lr=0.01):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []

    num_epochs = 100
    for epoch in range(num_epochs):    
        for x, y in train_loader:
            x = x.float()
            y = y.to(device)
            x = x.to(device)
            if flatten:
                x = torch.reshape(x, (x.shape[0], -1))
            prediction = model(x)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for x, y in test_loader:
            x = x.float()
            y = y.to(device)
            x = x.to(device)
            if flatten:
                x = torch.reshape(x, (x.shape[0], -1))
            prediction = model(x)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            y_all.append(y.cpu().detach().numpy())
            prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())
    
            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
    
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c


def run_test(data, e2e_lr, tnc_lr, cpc_lr, trip_lr, data_path, window_size, n_cross_val):
    # Load data
    with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(data_path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(data_path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    tnc_accs, tnc_aucs, tnc_auprcs = [], [], []
    for cv in range(n_cross_val):
        # Define baseline models
        if data == 'waveform':
            tnc_classifier = StateClassifier(input_size=3, output_size=8).to(device)
            tnc_classifier.train()
            tnc_model = torch.nn.Sequential(tnc_classifier).to(device)
        
        elif data == 'har':
            encoding_size = 10
            tnc_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            tnc_checkpoint = torch.load('./ckpt/har/checkpoint_%d.pth.tar'%cv)
            tnc_encoder.load_state_dict(tnc_checkpoint['encoder_state_dict'])
            tnc_classifier = StateClassifier(input_size=encoding_size, output_size=8).to(device)
            tnc_encoder.eval()
            tnc_classifier.train()
            tnc_model = torch.nn.Sequential(tnc_encoder, tnc_classifier).to(device)

        _, test_acc_tnc, test_auc_tnc, test_auprc_tnc, _ = epoch_run_encoder(tnc_model, train_loader, test_loader, flatten=True if data=='waveform' else False)
        tnc_accs.append(test_acc_tnc)
        tnc_aucs.append(test_auc_tnc)
        tnc_auprcs.append(test_auprc_tnc)

        with open("./outputs/%s_classifiers.txt"%data, "a") as f:
            f.write("\n\nPerformance result for a fold" )
            f.write("TNC model: \t AUC: %s\t Accuracy: %s \n\n" % (str(np.mean(tnc_accs)), str(100*np.mean(tnc_accs))))

        torch.cuda.empty_cache()

    print('=======> Performance Summary:')
    print('TNC model: \t Accuracy: %.2f +- %.2f \t AUC: %.3f +- %.3f \t AUPRC: %.3f +- %.3f'%
          (100 * np.mean(tnc_accs), 100 * np.std(tnc_accs), np.mean(tnc_aucs), np.std(tnc_aucs),
           np.mean(tnc_auprcs), np.std(tnc_auprcs)))


if __name__=='__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run classification test')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--data_path', type=str, default='./data/athena/Gesture/')
    parser.add_argument('--cv', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists('./ckpt/classifier_test'):
        os.mkdir('./ckpt/classifier_test')

    f = open("./outputs/%s_classifiers.txt"%args.data, "w")
    f.close()
    if args.data=='waveform':
        run_test(data='waveform', e2e_lr=0.0001, tnc_lr=0.01, cpc_lr=0.01, trip_lr=0.01,
                 data_path=args.data_path, window_size=4, n_cross_val=1)
    elif args.data=='har':
        run_test(data='har', e2e_lr=0.001, tnc_lr=0.1, cpc_lr=0.1, trip_lr=0.1,
                 data_path=args.data_path, window_size=4, n_cross_val=args.cv)
