import os
import re
import pandas as pd
from pathlib import Path

INPUT_ROOT = "data/raw/ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only"
OUTPUT_PATH = os.path.join("data", "words", "metadata.csv")  #OUTPUT_PATH = "data/words/metadata.csv"

MIN_SPEAKERS_PER_WORD = 2
MIN_REPS_PER_SPEAKER = 2


# (1) Clean the text and normailze a word
def clean_word(word: str) -> str:
    # handle non-string values and return empty string (like NaN or numbers)
    if not isinstance(word, str):
        return ""  

    word = word.lower()  # convert all to lower case
    word = word.replace("'", "") # remove apostrophes, since I am dealling with French
    # Keep only French letters
    word = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ]", "", word)
    return word.strip()


# (2) Convert a CSV files into a pandas DataFrame
def parse_words_csv(csv_path: Path) -> pd.DataFrame:
    # In the dataset I downloaded for this lab, it contains csv files that including transcriptions with timestamps.
    # CSV file structure = word, start_time, end_time
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=["word", "start", "end"]
    )

    # Remove quotes from word column and convert to string
    #         In the original CSV files, all words are quoted, so I need to remove the quotes.
    df["word"] = df["word"].astype(str).str.replace('"', '')
    df["word"] = df["word"].apply(clean_word)
    
    # Removeempty words
    df = df[df["word"] != ""]

    # convert timestamps to float (seconds)!
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)

    return df


# (3) Build the metadata
#     Walk through all speaker folders, find CSV files, and build the metadata table.
#     So basically, I only need two types of files, CSV and WAV.
#     each row = one word occurrence with audio and timestamps.
def build_metadata():
    records = []
    root = Path(INPUT_ROOT)

    if not root.exists():
        raise ValueError(f"ROOT path not found: {INPUT_ROOT}")

    # Loop though each speaker folder
    for speaker_dir in root.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker = speaker_dir.name # For example, the first five are AB, AB2, AG, AN, AR

        # Find all files ending with "_words.csv" in the speaker folder
        for csv_file in speaker_dir.glob("*_words.csv"):
            # extract utterance ID by removing "_words" from file name
            utt_id = csv_file.stem.replace("_words", "")
            # the corresponding audio should have the same name but ending with .wav
            wav_path = speaker_dir / f"{utt_id}.wav"

            if not wav_path.exists():  # skip if the audio is missing
                continue

            df = parse_words_csv(csv_file)

            # Create one record per word in this utterance
            for _, row in df.iterrows():
                records.append({
                    "audio_path": str(wav_path),
                    "start": row["start"],
                    "end": row["end"],
                    "duration": row["end"] - row["start"],  
                    "speaker": speaker,
                    "word": row["word"],
                    "utt_id": utt_id
                })

    meta = pd.DataFrame(records)

    if len(meta) == 0:
        raise ValueError("No data found. Check paths.")

    print(f"[INFO] Total segments before filtering: {len(meta)}")
    print(f"[INFO] Unique words before filtering: {meta['word'].nunique()}")

    return meta


# (4) Filter out the words for following analysis
def filter_dataset(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Since later I will do intra- and inter- speaker comparison, I need:
        Condition 1: Word must appear with at least MIN_SPEAKERS_PER_WORD different speakers
        Condition 2: Each speaker has at least MIN_REPS_PER_SPEAKER repetitions
    """
    valid_words = []

    for word, group in meta.groupby("word"): 
        speakers = group["speaker"].unique()

        # Condition 1: each word should have enough different speaker
        if len(speakers) < MIN_SPEAKERS_PER_WORD:
            continue

        # 条件2：each speaker should have enough repetitions
        ok = True
        for spk in speakers:
            if len(group[group["speaker"] == spk]) < MIN_REPS_PER_SPEAKER:
                ok = False
                break

        if ok:
            valid_words.append(word)

    filtered = meta[meta["word"].isin(valid_words)].copy()  # keep only words satisfying all conditions

    print(f"[INFO] Total segments after filtering: {len(filtered)}")
    print(f"[INFO] Unique words after filtering: {filtered['word'].nunique()}")

    return filtered


# (5) Save metadata to CSV file
def save_metadata(meta: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    meta.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Metadata saved to: {OUTPUT_PATH}")


def main():
    meta = build_metadata()
    meta = filter_dataset(meta)
    save_metadata(meta)


if __name__ == "__main__":
    main()