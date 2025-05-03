import argparse
import sys
from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset_loader import DatasetLoader
from bigram import BigramLanguageModel
from gpt import GPTLanguageModel

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def estimate_loss(model, dataset_loader, batch_size, block_size, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset_loader.get_batch(split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main(args):

    # load the data
    dataset_loader = DatasetLoader(args.data_file)

    # create the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == "bigram":
        model = BigramLanguageModel(
            dataset_loader.vocab_size, 
            embedding_dim=args.embedding_dim
        )
    elif args.model == "gpt":
        model = GPTLanguageModel(
            dataset_loader.vocab_size, 
            embedding_dim=args.embedding_dim, 
            block_size=args.block_size, 
            head_size=args.head_size, 
            dropout=args.dropout,
            num_heads=args.num_heads,
            num_layers=args.num_layers
        )
    model.to(device)

    # train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for iter in range(args.num_iterations):
        xb, yb = dataset_loader.get_batch("train", args.batch_size, args.block_size)
        xb = xb.to(device)
        yb = yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 1000 == 0:
            estimated_loss = estimate_loss(model, dataset_loader,args.batch_size, args.block_size, 100)
            logger.info(f"Iteration {iter} train loss: {estimated_loss['train']:.4f}, val loss: {estimated_loss['val']:.4f}")
    generated_text = model.generate(torch.zeros((1, args.context_length), dtype=torch.long), max_new_tokens=100)
    logger.info(f"Generated text: {dataset_loader.decode(generated_text[0].tolist())}")

        
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/input.txt")
    parser.add_argument("--model", type=str, choices=["bigram", "gpt"], default="bigram")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--head_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--context_length", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()
    main(args)