import os
import glob
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import mido
from state_spaces.src.models.s4 import S4  # Official S4 layer from HazyResearch
from transformers import AutoTokenizer, AutoModel
from einops import rearrange

LOG_DIR = os.getenv('LOG_DIR')
LOG_FILE = os.path.join(LOG_DIR, "model.log")
logger.add(LOG_FILE, rotation="500 MB")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load tokenizer for text embedding
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

def load_encodec_data(spec_dir, track_type):
    spec_files = glob.glob(os.path.join(spec_dir, "*", f"{track_type}_encodec.npy"))
    sequences = []
    for spec_file in spec_files:
        try:
            spec_data = np.load(spec_file, mmap_mode='r')  # Load the EnCodec data
            sequences.append(spec_data)
        except Exception as e:
            logger.error(f"Error processing {spec_file}: {e}")
    return np.array(sequences, dtype=np.float32)

def encode_text_prompt(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Shape: (1, 768)
    except Exception as e:
        logger.error(f"Error encoding text prompt: {e}")
        return np.zeros((1, 768))

class StripedHyenaLayer(nn.Module):
    def __init__(self, d_model, kernel_sizes=(3, 7, 15), dilation_rates=(1, 2, 4)):
        super(StripedHyenaLayer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding='same', dilation=d)
            for k, d in zip(kernel_sizes, dilation_rates)
        ])
        self.gate = nn.Conv1d(d_model, d_model * len(kernel_sizes), 1, padding='same')
        self.s4 = S4(d_model=len(kernel_sizes) * d_model, d_state=64, channels=1)
        self.residual = nn.Conv1d(d_model, d_model * len(kernel_sizes), 1, padding='same')
        self.norm = nn.LayerNorm(d_model * len(kernel_sizes))

    def forward(self, x):
        x = rearrange(x, 'b l d -> b d l')  # (batch, dim, length)
        convs = [conv(x) for conv in self.convs]
        combined_conv = torch.cat(convs, dim=1)
        gate = torch.sigmoid(self.gate(x))
        gated_conv = combined_conv * gate
        s4_out = self.s4(gated_conv)
        residual = self.residual(x)
        out = s4_out + residual
        out = rearrange(out, 'b d l -> b l d')
        out = self.norm(out)
        return out

class HierarchicalMusicGenerator(nn.Module):
    def __init__(self, num_tracks=4, text_dim=768, encodec_dim=60, sequence_length=500):
        super(HierarchicalMusicGenerator, self).__init__()
        self.num_tracks = num_tracks
        self.text_dim = text_dim
        self.encodec_dim = encodec_dim
        self.sequence_length = sequence_length

        # Structure Generator (Transformer)
        self.structure_dense = nn.Linear(text_dim, 128)
        self.structure_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.structure_output = nn.Linear(128, 7)  # 7 sections: Intro, Verse, etc.

        # Phrase Generator (StripedHyena)
        self.phrase_hyena = StripedHyenaLayer(d_model=encodec_dim * num_tracks + text_dim)
        self.phrase_output = nn.Linear(encodec_dim * num_tracks + text_dim, encodec_dim * num_tracks)

        # Encodec Generator (StripedHyena)
        self.encodec_hyena = StripedHyenaLayer(d_model=encodec_dim * num_tracks + text_dim)
        self.encodec_output = nn.Linear(encodec_dim * num_tracks + text_dim, encodec_dim * num_tracks)

    def forward(self, text_input, phrase_input, encodec_input):
        # Structure Generator
        structure_x = torch.relu(self.structure_dense(text_input))
        structure_x = structure_x.unsqueeze(0)  # Add sequence dim for attention
        structure_x, _ = self.structure_attention(structure_x, structure_x, structure_x)
        structure_x = structure_x.squeeze(0)
        structure_output = torch.softmax(self.structure_output(structure_x), dim=-1)

        # Phrase Generator
        phrase_input_combined = torch.cat([phrase_input, text_input.unsqueeze(1).repeat(1, phrase_input.size(1), 1)], dim=-1)
        phrase_x = self.phrase_hyena(phrase_input_combined)
        phrase_output = self.phrase_output(phrase_x)

        # Encodec Generator
        encodec_input_combined = torch.cat([encodec_input, text_input.unsqueeze(1).repeat(1, encodec_input.size(1), 1)], dim=-1)
        encodec_x = self.encodec_hyena(encodec_input_combined)
        encodec_output = self.encodec_output(encodec_x)

        return structure_output, phrase_output, encodec_output

def train_model(model, epochs=50, batch_size=64, checkpoint_dir="$CHECKPOINTS_DIR"):
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        track_types = ["vocals", "drums", "bass", "other"]
        spec_dir = os.getenv('SPEC_DIR')

        # Load Encodec data
        encodec_data = np.concatenate([load_encodec_data(spec_dir, t) for t in track_types], axis=0)

        # Encode text prompts
        text_data = np.array([encode_text_prompt("happy pop song") for _ in range(len(encodec_data))])

        # Check for data existence
        if len(encodec_data) == 0:
            logger.error("No Encodec data found. Verify preprocessing.")
            exit(1)

        # Convert data to tensors
        encodec_tensor = torch.tensor(encodec_data, dtype=torch.float32).to(device)
        text_tensor = torch.tensor(text_data, dtype=torch.float32).to(device)

        dataset_size = len(encodec_data)
        num_chunks = encodec_tensor.shape[1]  # Number of 500-frame chunks

        for epoch in range(epochs):
            model.train()
            for i in range(0, dataset_size, batch_size):
                batch_encodec = encodec_tensor[i:i+batch_size]
                batch_text = text_tensor[i:i+batch_size]

                # Pad the batch data with zeros if it is shorter than batch size
                if batch_encodec.shape[0] < batch_size:
                    padding_size = batch_size - batch_encodec.shape[0]
                    batch_encodec = torch.cat([batch_encodec, torch.zeros((padding_size, num_chunks, 60), dtype=torch.float32).to(device)], dim=0)
                    batch_text = torch.cat([batch_text, torch.zeros((padding_size, 768), dtype=torch.float32).to(device)], dim=0)

                # Split the batch into chunks along the sequence length dimension
                for j in range(num_chunks):
                    # Extract the current chunk
                    encodec_chunk = batch_encodec[:, j, :]
                    phrase_input = encodec_chunk  # Assign the chunk to phrase input

                    # Perform forward pass
                    _, phrase_output, encodec_output = model(batch_text, phrase_input, encodec_chunk)

                    # Compute loss
                    loss_phrase = criterion(phrase_output, phrase_input)
                    loss_encodec = criterion(encodec_output, encodec_chunk)
                    loss = loss_phrase + loss_encodec

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Log the training information
                    logger.info(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Chunk {j+1}, Loss: {loss.item()}")

            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt"))

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "music_generator.pt"))
        logger.success("Hierarchical model training completed")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    model = HierarchicalMusicGenerator().to(device)
    train_model(model)
