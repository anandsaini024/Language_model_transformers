import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simplified Transformer Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, attn_mask=mask)[0]
        query = self.dropout(self.norm1(attention + x))
        out = self.feed_forward(query)
        out = self.dropout(self.norm2(out + query))
        return out

# Define a simplified Decoder-Only Transformer
class TransformerDecoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_size, 
                 num_layers, 
                 heads, 
                 device, 
                 forward_expansion,
                 dropout,
                 max_length):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_size, 
                    heads, 
                    dropout=dropout, 
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, mask)

        out = self.fc_out(out)
        return out

# Example hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 10000  # size of vocabulary
embed_size = 256    # embedding size
num_layers = 4      # number of transformer layers
heads = 8           # number of heads in Multi-head Attention
forward_expansion = 4
dropout = 0.1
max_length = 100    # maximum length of a sequence

# Create the model
model = TransformerDecoder(vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)

model
