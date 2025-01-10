import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):
    
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        train_loss = 0.0
        for gene_tokens, protein_tokens, labels in train_loader:

            print('\nTraining')
            optimizer.zero_grad()

            gene_tokens = gene_tokens.to(device)
            protein_tokens = protein_tokens.to(device)
            labels = labels.to(device)

            outputs = model(gene_tokens, protein_tokens)
            loss_value = loss(outputs, labels)

            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gene_tokens, protein_tokens, labels in val_loader:

                gene_tokens = gene_tokens.to(device)
                protein_tokens = protein_tokens.to(device)
                labels = labels.to(device)

                outputs = model(gene_tokens, protein_tokens)
                loss_value = loss(outputs, labels)
                val_loss += loss_value.item()

            val_loss /= len(val_loader)
        val_losses.append(val_loss)

    return train_losses, val_losses


def get_predictions(model, val_loader, device, label_to_id, accuracy = True):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for gene_tokens, protein_tokens, labels in val_loader:
            gene_tokens = gene_tokens.to(device)
            protein_tokens = protein_tokens.to(device)
            labels = labels.to(device)

            outputs = model(gene_tokens, protein_tokens)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    predictions = torch.cat(predictions)
    true_labels = torch.cat(true_labels)

    pred_labels = [label_to_id[p.item()] for p in predictions]
    true_labels = [label_to_id[t.item()] for t in true_labels]

    if accuracy:
        accuracy = (pred_labels == true_labels).sum().item() / len(true_labels)
    else:
        accuracy = None

    return pred_labels, true_labels, accuracy


###################### Previous Functions ######################
def train_autoencoder(model, train_loader, val_loader, loss, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss_value = loss(reconstructed, x)
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x[0].to(device)
                reconstructed = model(x)
                loss_value = loss(reconstructed, x)
                val_loss += loss_value.item()


        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    return train_losses, val_losses


def old_get_predictions(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X_tensor)
        probs = F.softmax(y_pred, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy()

def old_train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        train_loss = 0.0
        x,y = next(iter(train_loader))
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss_value = loss(outputs, y)
        loss_value.backward()
        optimizer.step()
        train_loss += loss_value.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            x,y = next(iter(val_loader))
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            loss_value = loss(outputs, y)
            val_loss += loss_value.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    return train_losses, val_losses
 