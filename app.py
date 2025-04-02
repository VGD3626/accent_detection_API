import json
import io
import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch.nn.functional as F  # For softmax

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and processor
MODEL_NAME = "vrund1346/wav2vec2_accent_classification_v2"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

ACCENTS = ['arabic', 'dutch', 'english', 'french', 'german', 'korean',
           'mandarin', 'portuguese', 'russian', 'spanish', 'turkish']


def process_audio(file):
    audio, _ = librosa.load(file, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    # Apply softmax to normalize logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Map probabilities to percentages
    accent_scores = [{"name": ACCENTS[i], "confidence": round(probabilities[i].item() * 100, 2)}
                     for i in range(len(ACCENTS))]

    # Get the predicted accent based on the highest probability
    predicted_index = probabilities.argmax().item()
    predicted_accent = ACCENTS[predicted_index]

    return {"predicted_accent": predicted_accent, "accent_scores": accent_scores}


@app.route('/v1/predict', methods=['POST'])
def predict_accent():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    result = process_audio(request.files['file'])
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
