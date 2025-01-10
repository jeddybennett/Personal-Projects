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

class cross_attention_layer(nn.Module):
    def __init__(self, embedding_dim, nhead = 8, dropout = 0.1):
        super(cross_attention_layer, self).__init__()
        self.gene_to_protein = nn.MultiheadAttention(embedding_dim, nhead, dropout=dropout)
        self.protein_to_gene = nn.MultiheadAttention(embedding_dim, nhead, dropout=dropout)

        self.norm_gene = nn.LayerNorm(embedding_dim)
        self.norm_protein = nn.LayerNorm(embedding_dim)

        self.ff_gene = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim)
        )
        self.ff_protein = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim)
        )

    def forward(self, gene_tokens, protein_tokens):
        gene_updated, _ = self.gene_to_protein(gene_tokens, protein_tokens, protein_tokens)

        gene_encoded = self.norm_gene(gene_tokens + gene_updated)
        gene_encoded = self.norm_gene(gene_encoded + self.ff_gene(gene_encoded))

        protein_updated, _ = self.protein_to_gene(protein_tokens, gene_encoded, gene_encoded)

        protein_encoded = self.norm_protein(protein_tokens + protein_updated)
        protein_encoded = self.norm_protein(protein_encoded + self.ff_protein(protein_encoded))

        return gene_encoded, protein_encoded
    

class omics_transformer(nn.Module):
    def __init__(self, embedding_dim, num_cancers, nhead = 8, num_layers = 10, dim_feedforward = 256, dropout = 0.1):
        super(omics_transformer, self).__init__()
        self.embedding_dim = embedding_dim

        gene_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)

        self.gene_encoder = nn.TransformerEncoder(gene_encoder_layer, num_layers=num_layers)

        protein_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                            nhead=nhead,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=dropout)

        self.protein_encoder = nn.TransformerEncoder(protein_encoder_layer, num_layers=num_layers)
        self.cross_attention = cross_attention_layer(embedding_dim, nhead, dropout)

        self.combine = nn.Linear(2*embedding_dim, embedding_dim)

        self.final = nn.Linear(embedding_dim, num_cancers)

    def forward(self, gene_tokens, protein_tokens):
        gene_tokens = gene_tokens.permute(1,0,2)
        protein_tokens = protein_tokens.permute(1,0,2)

        gene_encoded = self.gene_encoder(gene_tokens)
        protein_encoded = self.protein_encoder(protein_tokens)

        gene_cross, protein_cross = self.cross_attention(gene_encoded, protein_encoded)

        gene_pooled = torch.mean(gene_cross, dim=1)
        protein_pooled = torch.mean(protein_cross, dim=1)

        combined = torch.cat([gene_pooled, protein_pooled], dim = 1)
        combined = self.combine(combined)

        logits = self.final(combined)

        return logits



###################### Archived Models ######################
class autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(autoencoder, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class cancer_transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, nhead=4, num_layers=10, dim_feedforward=256, dropout=0.1):
        super(cancer_transformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embedding = nn.Linear(1, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        self.transformer = nn.TransformerEncoder(
                          nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                          num_layers=num_layers)
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.final(x)
        return x


class cancer_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cancer_mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 3*hidden_size),
            nn.ReLU(),
            nn.Linear(3*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x