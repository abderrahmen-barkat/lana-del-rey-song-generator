import pandas as pd
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, AdamW

# Load the lyrics dataset
data = pd.read_csv('/content/sample_data/lyrics.csv')
data = data.drop('Link', axis=1)
lyrics = data['Lyrics'].tolist()

# Load the pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode the lyrics as tokens
tokens = tokenizer.encode("\n".join(lyrics))

# Define the text dataset and data loader
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.examples = []
        for i in range(0, len(data)-block_size+1, block_size):
            self.examples.append(data[i:i+block_size])
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

block_size = 128
dataset = TextDataset(tokens, block_size)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load the pre-trained model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
model.train()
for epoch in range(5):
    for batch in dataloader:
        batch = batch
        outputs = model(batch, labels=batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Set the model to evaluation mode
model.eval()

# Generate text using a prompt
prompt = "Ride:"
generated = tokenizer.encode(prompt)
past_key_values = None

for i in range(100):
    output, past_key_values = model(torch.tensor([generated]), past_key_values=past_key_values)
    token = torch.argmax(torch.round(output[..., -1, :]), dim=1)
    generated += [token.item()]

    # Check if the end-of-sequence token was generated
    if token.item() == 50256:
        break

# Check if generated is not empty
if len(generated) > 0:
    # Create the context tensor from generated
    context = torch.tensor([generated])
    
    # Decode the generated sequence, excluding the end-of-sequence token
    generated_text = tokenizer.decode(generated[:-1])
    print(generated_text)
else:
    print("Error: generated sequence is empty")