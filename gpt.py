import argparse
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, input_embedding_dim, hidden_embedding_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(input_embedding_dim, hidden_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_embedding_dim, input_embedding_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size, dropout):
        """
        Args:
            embedding_dim: the dimension of the input embedding
            head_size: the dimension of the query, key, and value
            block_size: the size of the context window
            dropout: the dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = dropout
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * self.head_size ** -0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        weights = self.dropout(weights)

        out = weights @ self.value(x) # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size, dropout, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(embedding_dim, head_size // num_heads, block_size, dropout) for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size, head_size)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size, dropout, num_heads):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_size, block_size, dropout, num_heads)
        self.feed_forward = FeedForward(head_size, 4*embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(head_size)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x)) # (B,T,head_size)
        x = x + self.feed_forward(self.ln2(x)) # (B,T,embedding_dim)
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, block_size, head_size, dropout, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.dropout = dropout
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        # self.multi_head_attention = MultiHeadAttention(embedding_dim, head_size, block_size, dropout, num_heads)
        # self.ln_f = nn.LayerNorm(embedding_dim) # final layer norm
        # self.feed_forward = FeedForward(head_size, embedding_dim, dropout)
        self.blocks = nn.Sequential(*[Block(embedding_dim, head_size, block_size, dropout, num_heads) for _ in range(num_layers)])
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.multi_head_attention(x) # (B,T,C)
        # x = self.ln_f(x) # (B,T,C)
        # x = self.feed_forward(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


    
    