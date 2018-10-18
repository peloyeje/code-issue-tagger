class ConvModel(torch.nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 vocab_size=10000,
                 seq_len=250):
        super(ConvModel,self).__init__()

        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._seq_len = seq_len

        self.embeddings = ScaledEmbedding(self._vocab_size, self._embedding_dim)
        self.conv = torch.nn.Conv1d(self._embedding_dim, 64, 5, padding=2)
        self.mp = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear(125 * 64, 2048)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, 1000)

    def forward(self, words_id):
        words_embedding = self.embeddings(words_id).permute(0,2,1)
        x = F.dropout(words_embedding, 0.2)
        x = self.conv(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.mp(x)
        x = x.view(-1, 125 * 64)
        x = self.fc1(x)
        x = F.dropout(x, 0.4)
        x = self.fc2(x)
        x = F.dropout(x, 0.4)
        x = self.fc3(x)
        return F.softmax(x)
