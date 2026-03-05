# EchoScript

EchoScript is an advanced audio captioning system that leverages deep learning to generate descriptive natural language captions for audio clips. It utilizes a hybrid architecture featuring an HTSAT encoder and a BART decoder to process complex audio environments and describe them through an immersive web interface.

---

## Features

### Deep Learning Powered
Uses a pre-trained HTSAT (Hierarchical Token-Semantic Audio Transformer) for robust audio feature extraction.

### Natural Language Generation
Employs a BART decoder to translate audio features into coherent, human-like captions.

### Real-Time Web Interface
Flask-based application featuring a cinematic landing page with 3D Spline integration.

### Beam Search Reasoning
Generates the top 3 candidate captions with probability scores to expose the model's decision-making process.

### Automated Audio Handling
Automatically manages resampling, mono conversion, and 10-second fixed-length padding or truncation.

---

## Architecture Overview

* **Encoder**: HTSAT (Hierarchical Token-Semantic Audio Transformer)
* **Decoder**: BART
* **Framework**: PyTorch
* **Inference Strategy**: Beam Search

---

## Tech Stack

### Backend
* Flask (Python)

### Deep Learning
* PyTorch
* Hugging Face Transformers

### Audio Processing
* Librosa

### Frontend
* HTML5 / CSS3 / JavaScript
* Spline (3D Integration)

---

### Project Structure

```
EchoScript/
│
├── app.py                      # Flask web server and API
├── caption_audio.py            # CLI script for audio captioning
├── inference_with_reasoning.py  # Script for detailed beam search analysis
├── models/                     # Model architecture and configurations
│   ├── bart_captioning_v2.py   # Core model implementation
│   └── audio_encoder.py        # HTSAT encoder logic
├── static/                     # Assets including CSS, JS, and images
├── templates/                  # HTML templates (landing.html, index.html)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
## Getting Started

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/PBAniket/EchoScript.git](https://github.com/PBAniket/EchoScript.git)
    cd EchoScript
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch librosa flask transformers werkzeug
    ```

### Running the Application

1.  **Web Interface**:
    Launch the Flask server:
    ```bash
    python app.py
    ```
    Open your browser and navigate to `http://127.0.0.1:5000` to access the interface.

2.  **CLI Inference**:
    Run the standalone script to caption local files:
    ```bash
    python caption_audio.py
    ```
    Enter the path to your audio file when prompted to receive the top 3 generated captions.

---

## Model Reasoning

EchoScript provides transparency in its decision-making by utilizing beam search. The system evaluates multiple potential sequences and ranks them based on their cumulative log-probability scores. The final output provided to the user is the sequence with the highest score.
