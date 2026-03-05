from flask import Flask, request, jsonify, render_template
import librosa
import torch
import torch.nn.functional as F
from models.bart_captioning_v2 import BartCaptionModelV2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

checkpoint_path = r"C:/Users/anike/OneDrive/Desktop/my files/langs/snake/notebooks/wavcaps/htsat_bart.pt"

print("Loading model checkpoint...")
model_loaded = False

cp = torch.load(checkpoint_path, map_location=torch.device("cpu"))
config = cp["config"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartCaptionModelV2(config)
model.load_state_dict(cp["model"])
model.to(device)
model.eval()
print("Model Ready ✓")
model_loaded = True


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/index")
def index_page():
    return render_template("index.html")


@app.route("/ready")
def ready():
    return jsonify({"ready": model_loaded})


@app.route("/upload", methods=["POST"])
def upload_audio():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file"}), 400

        filename = secure_filename(file.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(audio_path)

        sr = config["audio_args"]["sr"]
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
        waveform = torch.tensor(waveform).float()

        max_len = sr * 10
        if len(waveform) > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = F.pad(waveform, (0, max_len - len(waveform)))

        waveform = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            results = model.generate(
                samples=waveform, num_beams=3, num_return_sequences=3
            )

        best_caption = results[0][0]
        reasoning_output = [
            {"rank": i + 1, "caption": c, "score": float(s)}
            for i, (c, s) in enumerate(results)
        ]

        print("Generated:", best_caption)

        return jsonify({"caption": best_caption, "reasoning": reasoning_output})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
