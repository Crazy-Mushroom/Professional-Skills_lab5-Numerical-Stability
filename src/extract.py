import os
import time
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


INPUT_PATH = "data/words/metadata.csv"
OUTPUT_DIR = "features"
MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000
BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


# (1) Load the Wav2Vec model and its tokenizer (processor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model.to(device) # move model to CPU/GPT(if available) and set to evaluation mode (no training!)
model.eval()


# (2) Load a specific time segment from an audio file, and return the numpy array of audio samples.
def load_segment(path, start, end):       # star = start time in second, end = end time in second (to locate the audio sample)
    y, sr = librosa.load(path, sr=16000)  # automatically resample to 16kHz
    
    # convert seconds to sample indices
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    
    # cut out just the word segment
    segment = y[start_sample:end_sample]

    return segment


# (3) Extrat features (embeddins stored in float64 as baseline reference for comparisons later)
def extract():
    df = pd.read_csv(INPUT_PATH)

    embeddings = []

    start_time = time.time()

    for i in tqdm(range(0, len(df), BATCH_SIZE)): # process in batch for efficiency
        batch = df.iloc[i:i + BATCH_SIZE]    # get one batch of rows

        audio_list = []

        # load audio segments for this batch
        for _, row in batch.iterrows():
            segment = load_segment(
                row["audio_path"],
                row["start"],
                row["end"]
            )

            audio_list.append(segment)
        
        # convert audio to format wav2vec!
        inputs = processor(
            audio_list,
            sampling_rate=TARGET_SR,
            return_tensors="pt",  # return pytorch tensors
            padding=True
        )
        
        # move to CPU/GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():   # run the model without gradients, inference only
            outputs = model(**inputs)
        
        # last hiddent shape of the outputs = (batch size, time steps, hidden dim)
        hidden_states = outputs.last_hidden_state  # (B, T, D)

        # Average over time dimension to get one vector per word.
        # This gives us a fixed-size representation of the word regardless of the word length
        pooled = hidden_states.mean(dim=1)  # (B, D)

        embeddings.append(pooled.cpu().numpy())  # store as numpy cause it's easier to work with than tensors

    embeddings = np.vstack(embeddings)   # combine all batches into one big array

    ### convert to float64 (baseline reference)!
    embeddings = embeddings.astype(np.float64)

    elapsed = time.time() - start_time

    np.save(os.path.join(OUTPUT_DIR, "embeddings_float64.npy"), embeddings)

    print(f"[INFO] Saved embeddings: {embeddings.shape}")
    print(f"[INFO] Time: {elapsed:.2f}s")


if __name__ == "__main__":
    extract()