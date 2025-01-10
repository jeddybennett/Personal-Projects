import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm

def tokenize(gene_data, protein_data, embedding_dim, batch_size = 32):
    gene_array = gene_data.to_numpy()
    protein_array = protein_data.to_numpy()

    gene_tensor = torch.tensor(gene_array, dtype=torch.float32)
    protein_tensor = torch.tensor(protein_array, dtype=torch.float32)

    num_samples, num_genes = gene_tensor.shape
    num_samples, num_proteins = protein_tensor.shape

    gene_embedding = nn.Embedding(num_genes, embedding_dim)
    protein_embedding = nn.Embedding(num_proteins, embedding_dim)
    value_encoder = nn.Linear(1, embedding_dim)

    for start in range(0, num_samples, batch_size):

        end = min(start + batch_size, num_samples)
        gene_batch = gene_data.iloc[start:end].to_numpy()
        protein_batch = protein_data.iloc[start:end].to_numpy()

        gene_tensor = torch.tensor(gene_batch, dtype=torch.float32)
        protein_tensor = torch.tensor(protein_batch, dtype=torch.float32)

        batch_size_actual = gene_tensor.size(0)

        gene_ids = torch.arange(num_genes).unsqueeze(0).repeat(batch_size_actual, 1).long()
        protein_ids = torch.arange(num_proteins).unsqueeze(0).repeat(batch_size_actual, 1).long()

        gene_id_embeddings = gene_embedding(gene_ids)
        protein_id_embeddings = protein_embedding(protein_ids)

        gene_values = gene_tensor.unsqueeze(-1)
        protein_values = protein_tensor.unsqueeze(-1)

        gene_value_embeddings = value_encoder(gene_values)
        protein_value_embeddings = value_encoder(protein_values)

        gene_tokens = gene_id_embeddings + gene_value_embeddings
        protein_tokens = protein_id_embeddings + protein_value_embeddings

        yield gene_tokens, protein_tokens


class omics_dataset(Dataset):

    def __init__(self, gene_data, protein_data, labels, gene_embedding, protein_embedding, value_encoder):
        self.gene_data = gene_data
        self.protein_data = protein_data
        self.labels = labels
        self.gene_embedding = gene_embedding
        self.protein_embedding = protein_embedding
        self.value_encoder = value_encoder
        self.num_genes = gene_data.shape[1]
        self.num_proteins = protein_data.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gene_row = self.gene_data[idx]
        protein_row = self.protein_data[idx] # shape (1, num_proteins)

        gene_tensor = torch.tensor(gene_row, dtype=torch.float32)     # (1, num_genes)
        protein_tensor = torch.tensor(protein_row, dtype=torch.float32)# (1, num_proteins)

        gene_ids = torch.arange(self.num_genes).unsqueeze(0).long()      # (1, num_genes)
        protein_ids = torch.arange(self.num_proteins).unsqueeze(0).long()# (1, num_proteins)

        gene_id_embeddings = self.gene_embedding(gene_ids)
        protein_id_embeddings = self.protein_embedding(protein_ids)

        gene_values = gene_tensor.unsqueeze(-1)    # (1, num_genes, 1)
        protein_values = protein_tensor.unsqueeze(-1) # (1, num_proteins, 1)

        gene_value_embeddings = self.value_encoder(gene_values)
        protein_value_embeddings = self.value_encoder(protein_values)

        gene_tokens = gene_id_embeddings + gene_value_embeddings   # (1, num_genes, embedding_dim)
        protein_tokens = protein_id_embeddings + protein_value_embeddings # (1, num_proteins, embedding_dim)

        # Squeeze out the batch dimension since we return a single sample
        gene_tokens = gene_tokens.squeeze(0)
        protein_tokens = protein_tokens.squeeze(0)

        return gene_tokens, protein_tokens, self.labels[idx]