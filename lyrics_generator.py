import os
import google.generativeai as genai
from loguru import logger

LYRICS_DIR = os.getenv('LYRICS_DIR')
MODEL_DIR = os.getenv('MODEL_DIR')
BASE_DIR = os.getenv('BASE_DIR')
LOG_DIR = os.getenv('LOG_DIR')
LOG_FILE = os.path.join(LOG_DIR, "lyrics_generator.log")

logger.add(LOG_FILE, rotation="500 MB")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    exit(1)
genai.configure(api_key=GEMINI_API_KEY)

def fine_tune_lyrics_model():
    try:
        dataset_path = os.path.join(BASE_DIR, "lyrics_dataset.jsonl")
        if not os.path.exists(dataset_path):
            logger.error("Lyrics dataset not found. Run preprocess.py first.")
            exit(1)

        base_model = "models/gemini-1.5-flash-001-tuning"
        operation = genai.create_tuned_model(
            source_model=base_model,
            training_data=dataset_path,
            id="lyrics-generator",
            epoch_count=100,
            batch_size=4,
            learning_rate=0.001,
        )
        logger.info("Fine-tuning started. Check Google AI Studio for status.")
        return operation
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        exit(1)

def generate_lyrics(prompt, max_length=200):
    try:
        model = genai.GenerativeModel("tunedModels/lyrics-generator")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating lyrics: {e}")
        return None

if __name__ == "__main__":
    operation = fine_tune_lyrics_model()
    # Test generation after fine-tuning
    lyrics = generate_lyrics("Write a verse for a happy pop song in C major")
    if lyrics:
        logger.info(f"Generated lyrics:\n{lyrics}")
