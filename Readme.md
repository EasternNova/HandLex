# 🤟 HandLex

### Real-Time Sign Language Recognition & Translation

**HandLex** is an AI-powered sign language recognition system designed to bridge the communication gap between sign-language users and non-signers.

The project uses **computer vision, deep learning, and natural language processing** to recognize hand gestures from video and convert them into meaningful words or text.

---

## 🚀 Project Goal

The main goal of HandLex is to build an accessible system that can:

**Camera / Video → Sign Language Gesture → AI Model → Recognized Word → Text**

The system is intended to recognize individual sign-language gestures and provide their corresponding words in real time.

---

## ✨ Planned Features

- 🤟 Sign language gesture recognition
- 🎥 Video-based gesture analysis
- 📷 Real-time webcam recognition
- 🧠 Deep-learning-based classification
- 🔤 Conversion of gestures into words
- ⚡ Real-time prediction
- 🌐 Simple web interface
- 📊 Model performance visualization
- 🔄 Support for expanding the vocabulary
- 📁 Dataset preprocessing and management

---

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Main backend & ML development |
| **PyTorch** | Deep learning model development |
| **Scikit-learn** | Data preprocessing, evaluation & traditional ML utilities |
| **OpenCV** | Video processing and computer vision |
| **NumPy** | Numerical computation |
| **Pandas** | Dataset and annotation handling |
| **Streamlit** | Interactive ML/web interface |
| **React + Vite** | Project website / frontend |
| **Git & GitHub** | Version control |
| **Jupyter Notebook** | Experimentation and model development |

---

## 📂 Project Structure

```text
HandLex/
│
├── data/
│   ├── raw/                 # Original datasets
│   ├── interim/             # Temporary processed data
│   ├── processed/           # Model-ready data
│   └── annotations/         # Labels and metadata
│
├── models/
│   ├── checkpoints/         # Training checkpoints
│   └── pretrained/          # Pretrained models
│
├── notebooks/               # Experiments and analysis
│
├── src/
│   ├── __init__.py
│   ├── data/                # Dataset loading & preprocessing
│   ├── features/            # Feature extraction
│   ├── models/              # Model architectures
│   ├── training/            # Training scripts
│   ├── inference/           # Prediction/inference
│   └── utils/               # Helper utilities
│
├── app/                     # Streamlit application
│
├── website/                 # React/Vite project website
│
├── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
```

---

# 📊 Dataset

HandLex is designed to work with publicly available sign-language datasets.

Potential datasets include:

- **WLASL** — Word-Level American Sign Language
- **How2Sign** — Large-scale continuous American Sign Language dataset
- Other appropriately licensed sign-language datasets

Dataset availability, licensing, and permitted usage will be checked before integrating any dataset into the final system.

### Dataset Pipeline

```text
Raw Dataset
     ↓
Data Validation
     ↓
Video Processing
     ↓
Frame Extraction
     ↓
Feature Extraction
     ↓
Train / Validation / Test Split
     ↓
Model Training
```

---

# 🏗️ System Architecture

```text
                ┌─────────────────┐
                │   Camera/Video  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │  Video Input    │
                │    OpenCV       │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Preprocessing   │
                │ & Frame Sampling│
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Feature / Pose  │
                │   Extraction    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Deep Learning   │
                │     Model       │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │   Prediction    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Recognized Word │
                └─────────────────┘
```

---

# 🔬 Machine Learning Pipeline

### 1. Data Collection

Collect videos representing different sign-language words.

### 2. Preprocessing

Videos are standardized by:

- Sampling frames
- Resizing frames
- Normalizing input
- Handling variable video lengths
- Removing invalid samples

### 3. Feature Extraction

Relevant visual information can be extracted from:

- Hand movements
- Hand position
- Body pose
- Facial information
- Temporal movement

### 4. Model Training

The processed sequences are provided to a deep-learning model to learn the relationship between:

```text
Visual Sequence → Sign → Word
```

### 5. Evaluation

The model will be evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 6. Inference

The trained model receives a new video/webcam sequence and predicts the corresponding sign.

---

# 🖥️ Application

The planned HandLex interface will allow users to interact with the trained model through a simple web interface.

Possible workflow:

```text
Open HandLex
     ↓
Enable Camera
     ↓
Perform Sign
     ↓
Capture Video Sequence
     ↓
AI Processes Gesture
     ↓
Prediction
     ↓
Display Recognized Word
```

The ML application can be developed using **Streamlit**, while the main project website is developed using **React + Vite**.

---

# 🧪 Development Workflow

```text
Dataset
   ↓
Exploration
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Development
   ↓
Training
   ↓
Evaluation
   ↓
Optimization
   ↓
Inference
   ↓
Streamlit Application
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd HandLex
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running HandLex

### Run the Streamlit application

```bash
streamlit run app/app.py
```

### Run the website

From the website directory:

```bash
npm install
npm run dev
```

---

# 📈 Current Development Status

> 🚧 **HandLex is currently under active development.**

### Completed / In Progress

- [x] Initial project structure
- [x] Git repository setup
- [x] Website initialization with React + Vite
- [x] Initial HandLex website design
- [ ] Dataset finalization
- [ ] Dataset preprocessing pipeline
- [ ] Feature extraction
- [ ] Baseline model
- [ ] Model training
- [ ] Model evaluation
- [ ] Real-time inference
- [ ] Streamlit application
- [ ] Website ↔ ML integration
- [ ] Final deployment

---

# 🎯 Future Improvements

Future versions of HandLex may include:

- Continuous sign-language recognition
- Sentence generation
- Text-to-speech output
- Speech-to-text input
- Larger sign vocabulary
- Multiple sign languages
- Improved real-time performance
- Mobile support
- Cloud deployment
- Personalized recognition
- Better handling of different lighting and backgrounds

---

# 🤝 Contribution

Contributions are welcome.

If you would like to contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Test your implementation
5. Submit a pull request

---

# 📜 License

The HandLex source code will be released under an appropriate open-source license.

**Dataset licenses remain separate and must be respected according to their individual terms.**

---

# 👩‍💻 Project

**HandLex**  
*Turning signs into words. Connecting communication through AI.*

> 🤟 **See the sign. Understand the meaning.**