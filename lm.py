import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# --- Transformer Model Definition (from your initial code) ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, attn_mask=mask)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out


# --- Dataset Class ---

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):
        self.block_size = config["block_size"]
        
        # Unique characters and mapping to integers
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode the entire data to integers
        self.data = [self.stoi[ch] for ch in data]

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        # -1 because we need to return the next character as the label
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]

        # Convert to tensors
        input_tensor = torch.tensor(chunk[:-1], dtype=torch.long)
        target_tensor = torch.tensor(chunk[1:], dtype=torch.long)

        return input_tensor, target_tensor

# Example of usage
config = {
    "block_size": 128  # Sequence length
}

file_path = "C:/Users/shash/Documents/GitHub/Language_model_transformers/half_data.txt"  # Update with your file path

with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

config = {"block_size": 128}
dataset = CharDataset(config, data)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# --- Model Initialization ---
vocab_size = len(dataset.stoi)  # Vocabulary size
embed_size = 768
num_layers = 12
heads = 8
forward_expansion = 4
dropout = 0.1
max_length = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Decoder(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_length=max_length,
).to(device)

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# --- Training Loop ---
for epoch in range(num_epochs):
    for batch, (input_seq, target_seq) in enumerate(data_loader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        optimizer.zero_grad()
        output = model(input_seq, None)
        loss = criterion(output.transpose(1, 2), target_seq)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch+1}/{len(data_loader)}], Loss: {loss.item()}")

# --- Save the Model ---
torch.save(model.state_dict(), 'C:/Users/shash/Documents/GitHub/Language_model_transformers/model.pth')

print("Training Complete")

# --- Load the Trained Model ---
model_path = 'C:/Users/shash/Documents/GitHub/Language_model_transformers/model.pth'  # Path to your saved model
model.load_state_dict(torch.load(model_path))
model.eval()

# --- Function to Generate Text ---
def generate_text(model, initial_str, gen_size=100, temperature=1.0):
    model.eval()
    generated_text = initial_str
    sequence = [dataset.stoi[c] for c in initial_str]

    for _ in range(gen_size):
        input_sequence = torch.tensor(sequence[-max_length:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_sequence, None)
        
        output = output[:, -1, :]
        probabilities = torch.softmax(output / temperature, dim=-1).squeeze()
        
        next_char_idx = torch.multinomial(probabilities, 1).item()
        sequence.append(next_char_idx)

        generated_text += dataset.itos[next_char_idx]

    return generated_text

# --- Generate Text from a Prompt ---
prompt = "Your prompt here"  # Replace with your starting string
generated_text = generate_text(model, prompt, gen_size=200)  # Generate 200 characters

print(generated_text)