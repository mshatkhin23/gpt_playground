import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}
        self._load_data()

    def _load_data(self):
        # load in the dataset
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()
        logger.info(f"Loaded {len(text)} characters from {self.file_path}")

        # calculate the vocab size
        unique_chars = sorted(list(set(text)))
        self.vocab_size = len(unique_chars)
        self.char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        logger.info(f"Vocab size: {self.vocab_size}")

        # encode and split the data into train and validation sets
        self.encode = lambda s: [self.char_to_index[c] for c in s]
        self.decode = lambda l: "".join([self.index_to_char[i] for i in l])
        self.data["all"] = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(self.data["all"]))
        self.data["train"] = self.data["all"][:n]
        self.data["val"] = self.data["all"][n:]
        logger.info(
            f"Train data size: {len(self.data['train'])}, Val data size: {len(self.data['val'])}"
        )
        return self.data["train"], self.data["val"]

    def get_batch(self, split, batch_size, block_size):
        # Generate indices on CPU first
        batch_data = torch.randint(0, len(self.data[split]) - block_size, (batch_size,), device='cpu')
        # Stack tensors on CPU
        x = torch.stack([self.data[split][i : i + block_size] for i in batch_data])
        y = torch.stack(
            [self.data[split][i + 1 : i + block_size + 1] for i in batch_data]
        )
        return x, y
