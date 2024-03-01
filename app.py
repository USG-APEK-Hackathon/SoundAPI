from flask import Flask, request, jsonify
from openai import OpenAI
import logging
from transformers import pipeline
import os

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

emotion_classifier = pipeline("text-classification",
                              model='bhadresh-savani/distilbert-base-uncased-emotion')


def classify(audio_path):
    client = OpenAI()

    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    ).text
    emotion = emotion_classifier(transcription)
    detected_emotion = emotion
    logging.info(f"Transcription: {transcription}")
    return detected_emotion


@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        audio_file = request.files['audio']
        audio_file.save('temp_audio.mp3')
        result = classify('temp_audio.mp3')
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in processing audio: {str(e)}")
        return jsonify({"error": "Error in processing audio"}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))

    is_prod = os.environ.get('RAILWAY_ENVIRONMENT_NAME') is not None

    app.run(host='0.0.0.0', port=port, debug=not is_prod)