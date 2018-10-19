import torch
import torch.nn.functional as F

class ScaledEmbedding(torch.nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ConvModel(torch.nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 vocab_size=10000,
                 seq_len=250):
        super(ConvModel,self).__init__()

        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._seq_len = seq_len

        self._max_pool_kernel = 2

        self.embeddings = ScaledEmbedding(self._vocab_size, self._embedding_dim)
        self.conv1 = torch.nn.Conv1d(self._embedding_dim, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 32, 5, padding=2)
        self.mp1 = torch.nn.MaxPool1d(self._max_pool_kernel)
        self.mp2 = torch.nn.MaxPool1d(self._max_pool_kernel)
        self.fc1 = torch.nn.Linear(62 * 32, 1024)
        self.fc2 = torch.nn.Linear(1024, 100)

    def forward(self, x):
        x = self.embeddings(x).permute(0,2,1)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.mp1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.mp2(x)
        x = x.reshape(-1, 62 * 32)
        x = self.fc1(x)
        x = F.dropout(x, 0.2)
        x = self.fc2(x)
        return x
