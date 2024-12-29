import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math

class GINConv(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, A, X):
		X = self.linear(X + A @ X)
		X = torch.nn.functional.relu(X)
		return X


class TGAE_Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
	        super(TGAE_Encoder, self).__init__()
	        if isinstance(hidden_dim, int):
	            hidden_dim = [hidden_dim] * num_hidden_layers  # Repeat the same size for all layers
	
	        self.in_proj = torch.nn.Linear(input_dim, hidden_dim[0])
	        self.hidden_layers = nn.ModuleList([
	            torch.nn.Linear(hidden_dim[i], hidden_dim[i + 1]) for i in range(len(hidden_dim) - 1)
	        ])
	        self.out_proj = torch.nn.Linear(hidden_dim[-1], output_dim)


	def forward(self, A, X):
		initial_X = torch.empty_like(X).copy_(X)
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			X = layer(A, torch.cat([initial_X,X],dim=1))
			hidden_states.append(X)
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X

class TGAE(torch.nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers+2)

	def forward(self, X, adj):
		Z = self.encoder(adj, X)
		return Z

