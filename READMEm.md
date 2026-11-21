# HandLex-hhhh

> Exploring sign-language hand-sign recognition using computer vision and machine learning.

## 1. Project Overview

**HandLex** is a sign-language recognition and translation project exploring how computer vision and machine learning can help bridge the communication gap between sign-language users and people unfamiliar with sign language.

HandLex uses hand-landmark detection and machine learning to recognize hand signs from visual input. It extracts hand-related features, processes them through a trained recognition model, and presents recognition results through a web-based interface.

The project combines a Python machine-learning backend with a React frontend. Its current verified recognition scope is **static ASL alphabet gesture classification**, rather than full continuous sign-language translation.

## 2. Project Objectives

- Explore computer vision and machine learning for hand-sign recognition.
- Detect hand landmarks and convert them into numerical features.
- Train and evaluate a static hand-sign classification model.
- Provide a web interface and backend API for interacting with the recognition pipeline.
- Explore the integration of machine-learning inference into a web application.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Machine-learning development, feature extraction, training, inference and backend logic |
| FastAPI | Backend API and endpoints |
| Uvicorn | Runs the FastAPI application |
| React | Frontend interface |
| Vite | Frontend development server and build tool |
| MediaPipe | Hand-landmark detection |
| OpenCV | Image processing and visual-input handling |
| NumPy | Numerical processing and feature arrays |
| Scikit-learn | Model A classification and machine-learning operations |
| Joblib | Saving and loading the trained classifier |
| Pillow (PIL) | Image handling and processing |
| Python Multipart | Multipart form-data handling for uploads |
| XGBoost | Included in the dependency list for the broader machine-learning environment and experiments; not identified as part of the active Model A inference pipeline |

## 4. Repository Structure

```text
HandLex/
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   └── model_a_service.py
│   ├── modelA/
│   │   ├── features.py
│   │   ├── inference.py
│   │   ├── landmarker.py
│   │   ├── model.py
│   │   ├── prediction_stabilizer.py
│   │   ├── prepare_data_v6.py
│   │   ├── train.py
│   │   ├── webcam.py
│   │   ├── test_hand_landmarker.py
│   │   ├── artifacts/
│   │   ├── checkpoints/
│   │   └── data/
│   └── models/
│       └── pretrained/
│           └── hand_landmarker.task
├── dataset/
│   └── ASL/
│       ├── asl_alphabet_train/
│       │   └── asl_alphabet_train/
│       │       ├── A/
│       │       ├── B/
│       │       ├── ...
│       │       ├── Z/
│       │       ├── del/
│       │       ├── nothing/
│       │       └── space/
│       └── asl_alphabet_test/
│           ├── A_test.jpg
│           ├── B_test.jpg
│           ├── ...
│           └── Z_test.jpg
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── pages/
│   │   ├── styles/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── .gitignore
├── requirements.txt
├── README.md
└── ...
```

### Directory overview

| Location | Purpose |
|---|---|
| `backend/api/` | FastAPI routes and service layer |
| `backend/modelA/` | Model A feature extraction, training, inference, landmark processing and stabilization |
| `backend/modelA/artifacts/` | Trained model artifacts and associated metadata |
| `backend/modelA/checkpoints/` | Training or experimental checkpoints |
| `backend/modelA/data/` | Prepared and processed data for Model A |
| `backend/models/pretrained/` | Pretrained hand-landmark resources |
| `dataset/ASL/` | ASL Alphabet image data |
| `frontend/src/` | React application source, pages and styles |
| `frontend/public/` | Static frontend assets |
| `frontend/package.json` | Frontend dependencies and scripts |
| `requirements.txt` | Python dependencies |

Generated and temporary files such as `node_modules`, `dist`, `__pycache__`, virtual environments and other build files are omitted from the repository tree.

## 5. Dataset & Data Preparation

### 5.1 Dataset

Model A uses an ASL Alphabet image dataset containing images of hand gestures representing letters of the American Sign Language alphabet, along with the additional categories `del`, `nothing` and `space`.

> **Dataset attribution and licensing are pending verification.** Before redistributing the dataset or publishing definitive attribution, confirm the original source, creator and license terms.

### 5.2 Dataset structure

```text
dataset/
└── ASL/
    ├── asl_alphabet_train/
    │   └── asl_alphabet_train/
    │       ├── A/
    │       ├── B/
    │       ├── ...
    │       ├── Z/
    │       ├── del/
    │       ├── nothing/
    │       └── space/
    └── asl_alphabet_test/
        ├── A_test.jpg
        ├── B_test.jpg
        ├── ...
        ├── Z_test.jpg
        ├── del_test.jpg
        ├── nothing_test.jpg
        └── space_test.jpg
```

### 5.3 Classes

Model A classifies 29 labels:

- **26 alphabet classes:** `A`–`Z`
- **3 additional classes:** `del`, `nothing` and `space`

### 5.4 Preprocessing

The Model A pipeline uses landmark-derived features rather than classifying the raw image pixels directly:

1. Load the input image.
2. Detect hand landmarks using MediaPipe Hand Landmarker.
3. Extract the hand landmarks.
4. Convert landmark information into numerical features.
5. Prepare the feature vector for classification.

### 5.5 Train/test split

The reported Model A V6 evaluation used a stratified 80/20 train/test split:

| Parameter | Value |
|---|---|
| Training split | 80% |
| Testing split | 20% |
| Splitting method | Stratified |
| Random state | `42` |
| Test samples | `13,318` |

Stratification was used to maintain class representation across the train and test subsets.

## 6. Frontend — HandLex Web Interface

HandLex uses **React** and **Vite** for its frontend.

The frontend provides the web interface, displays project information, presents the translation interface and communicates with the backend through HTTP requests.

- **React:** UI components and application interface.
- **Vite:** Frontend development server and build tool.
- **Home page:** Project introduction and navigation.
- **Translate page:** Interface for interacting with sign-recognition functionality.

The complete frontend-to-backend continuous webcam workflow has not been fully verified.

### Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Vite normally serves the development frontend at:

`http://localhost:5173`

## 7. Backend & Machine Learning

### 7.1 FastAPI backend

FastAPI provides endpoints for backend status, health checks, Model A management and image-based prediction.

The API routing and request handling are separated from the model inference logic through a dedicated Model A service.

#### Backend components

| Component | File | Responsibility |
|---|---|---|
| API application | `backend/api/main.py` | API routes, health checks, model operations and image prediction |
| Model service | `backend/api/model_a_service.py` | Model A operations and inference-engine connection |
| Inference engine | `backend/modelA/inference.py` | Recognition pipeline and classification |
| Feature extraction | `backend/modelA/features.py` | Converts hand landmarks into numerical features |
| Hand-landmark detection | `backend/modelA/landmarker.py` | MediaPipe landmark detection |
| Prediction stabilization | `backend/modelA/prediction_stabilizer.py` | Stabilizes repeated predictions |

#### Inference workflow

1. An image is sent to the backend.
2. FastAPI receives the request.
3. The image is decoded.
4. MediaPipe Hand Landmarker detects and extracts hand landmarks.
5. The landmarks are converted into a feature vector.
6. Model A classifies the feature vector.
7. Stabilization is applied where configured.
8. The prediction is returned through the API.

#### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API information or status |
| `GET` | `/health` | Backend health check |
| `GET` | `/models/model-a/status` | Model A status and resource information |
| `POST` | `/models/model-a/load` | Load Model A |
| `POST` | `/models/model-a/unload` | Unload Model A |
| `POST` | `/models/model-a/predict-image` | Image-based prediction |

See [API Reference](#10-api-reference) for details and testing instructions.

#### API documentation

FastAPI generates interactive API documentation:

`http://localhost:8000/docs`

The Swagger interface can be used to inspect endpoints, view request requirements and test API operations.

#### CORS

The backend development configuration currently allows these frontend origins:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

### 7.2 Model A — Static Sign Recognition

Model A is the currently documented recognition model in HandLex. It classifies static hand gestures corresponding to ASL alphabet letters and three additional classes.

**Model A V6 uses a Random Forest classifier and Feature Version 3 (V3).**

#### Recognition approach

1. Receive an image containing a hand gesture.
2. Detect the hand with MediaPipe Hand Landmarker.
3. Extract the 21 hand landmarks.
4. Transform the landmarks into a numerical feature vector.
5. Pass the feature vector to the trained classifier.
6. Obtain a predicted class.
7. Apply prediction stabilization where configured.

#### Algorithm

Model A V6 uses a **Random Forest Classifier**, an ensemble algorithm that combines decision trees for classification.

#### Feature extraction — V3

The V3 feature pipeline generates **95 features**:

| Feature group | Count | Description |
|---|---:|---|
| Normalized XYZ coordinates | 63 | XYZ coordinates for 21 hand landmarks |
| Joint-angle features | 10 | Geometric finger and joint relationships |
| Wrist-distance features | 20 | Distance measurements relative to the wrist |
| Hand-presence indicator | 1 | Hand-presence information |
| Handedness indicator | 1 | Handedness information |
| **Total** | **95** | Complete feature vector |

#### Classification labels

The model recognizes 29 classes:

`A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`

Additional classes:

| Label | Description |
|---|---|
| `del` | Deletion-related gesture category |
| `nothing` | No-active-sign category |
| `space` | Space-related gesture category |

#### Training workflow

1. Prepare the dataset.
2. Prepare numerical feature representations.
3. Associate features with class labels.
4. Split the dataset into training and testing subsets.
5. Train the Random Forest classifier.
6. Evaluate the trained classifier.
7. Save the model artifact.

#### Model artifact

The trained Model A V6 artifact is expected at:

`backend/modelA/artifacts/modelA_rf_v6.pkl`

The inference pipeline uses the artifact and associated metadata to load the trained classifier and expected feature configuration.

> The model artifact is approximately 208 MB. Confirm how it is distributed before cloning or downloading the repository. If it is not included in the repository, arrange a separate download/setup step. GitHub's standard file-size limits may require Git LFS or external hosting.

#### Evaluation

| Parameter | Value |
|---|---|
| Model | Model A V6 |
| Algorithm | Random Forest Classifier |
| Feature version | V3 |
| Features | 95 |
| Classes | 29 |
| Training split | 80% |
| Testing split | 20% |
| Split method | Stratified |
| Random state | `42` |
| Test samples | 13,318 |
| **Reported accuracy** | **98.8287%** |

The reported accuracy is the result of the documented test evaluation. It is not a guarantee of performance in real-world signing conditions.

#### Prediction stabilization

Model A includes a stabilization component to reduce fluctuations across repeated predictions.

| Parameter | Value |
|---|---:|
| History size | 7 |
| Stable prediction count | 5 |
| Confidence threshold | 0.50 |

Stabilization is intended to improve consistency in repeated outputs. It does not establish that continuous sign-language translation is implemented.

#### Model A limitations

Model A currently focuses on static gesture classification. It does not inherently interpret:

- Temporal relationships between consecutive signs
- Full sign-language grammar
- Sentence structure
- Facial expressions
- Body movement and posture
- Context-dependent meaning

### 7.3 Pretrained resource — MediaPipe Hand Landmarker

| Resource | Location | Purpose |
|---|---|---|
| MediaPipe Hand Landmarker | `backend/models/pretrained/hand_landmarker.task` | Detects and extracts hand landmarks |
| HandLex Model A V6 | `backend/modelA/artifacts/modelA_rf_v6.pkl` | Classifies extracted features into 29 labels |

The pretrained Hand Landmarker and Model A classifier perform different tasks and are both required for the documented Model A image-recognition pipeline.

> Confirm how `hand_landmarker.task` is distributed in the repository and provide a setup/download step if it is not included.

## 8. Installation & Running HandLex

### 8.1 Prerequisites

| Software | Purpose |
|---|---|
| Python | Backend and machine-learning environment |
| Node.js | Frontend development environment |
| npm | Frontend dependency management |
| Git | Repository management (optional when downloading directly) |

### 8.2 Clone or download

Clone the repository or download its source files using your preferred method.

### 8.3 Create a Python virtual environment

From the project root:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is successful, the terminal should display the virtual-environment name.

### 8.4 Install Python dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

### 8.5 Verify required resources

Ensure the required files are available:

```text
backend/
├── modelA/
│   └── artifacts/
│       └── modelA_rf_v6.pkl
└── models/
    └── pretrained/
        └── hand_landmarker.task
```

If either file is not included in the repository, obtain it from the project's designated distribution source before starting model operations.

### 8.6 Start the backend

From the project root:

```powershell
cd backend
uvicorn api.main:app --reload
```

The backend normally runs at:

`http://localhost:8000`

### 8.7 Start the frontend

Open a second terminal from the project root:

```powershell
cd frontend
npm install
npm run dev
```

The frontend normally runs at:

`http://localhost:5173`

### 8.8 Local URLs

| Service | URL |
|---|---|
| HandLex frontend | `http://localhost:5173` |
| FastAPI backend | `http://localhost:8000` |
| API documentation | `http://localhost:8000/docs` |
| API root | `http://localhost:8000/` |

### 8.9 First-time setup checklist

- [ ] Install Python and Node.js.
- [ ] Clone or download HandLex.
- [ ] Create and activate the Python virtual environment.
- [ ] Install Python dependencies.
- [ ] Install frontend dependencies.
- [ ] Obtain the required MediaPipe Hand Landmarker resource.
- [ ] Obtain the trained Model A artifact.
- [ ] Start the backend.
- [ ] Start the frontend.
- [ ] Open the frontend in a browser.
- [ ] Use `/docs` to test backend operations if needed.

### 8.10 Shut down

To stop a running server, use its terminal and press `Ctrl + C`.

To deactivate the Python virtual environment:

```powershell
deactivate
```

## 9. API Reference

HandLex provides API endpoints for backend health checks, Model A management and image-based prediction.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `GET` | `/models/model-a/status` | Model A status |
| `POST` | `/models/model-a/load` | Load Model A |
| `POST` | `/models/model-a/unload` | Unload Model A |
| `POST` | `/models/model-a/predict-image` | Image-based prediction |

### Testing the endpoints

The interactive API documentation is available at:

`http://localhost:8000/docs`

Use Swagger to inspect the actual request fields, parameters and response schemas, and to execute test requests.

### Image prediction

`POST /models/model-a/predict-image`

This endpoint accepts an image through multipart/form-data and processes it through the Model A inference pipeline.

The exact upload field and response structure should be checked in the live API documentation (`/docs`) before relying on a manually constructed request.

## 10. Troubleshooting

### Python is not recognized

- Verify that Python is installed.
- Check that Python is available on the system PATH.
- Restart the terminal after PATH changes.

Check the installation:

```powershell
python --version
```

If needed, try:

```powershell
py --version
```

### Virtual-environment activation fails

PowerShell may restrict script execution. Review the execution policy for your environment or use a different terminal such as Command Prompt.

### Dependencies fail to install

- Confirm the Python version.
- Confirm the virtual environment is active.
- Check pip availability.
- Check internet connectivity.
- Check package compatibility.

Verify pip:

```powershell
python -m pip --version
```

### Backend does not start

- Check that the virtual environment is active.
- Confirm backend dependencies are installed.
- Run the startup command from the correct directory.
- Review terminal output for import errors.

From the project root:

```powershell
cd backend
uvicorn api.main:app --reload
```

### Port already in use

Stop the other process using the port or run the backend on another available port, for example:

```powershell
uvicorn api.main:app --reload --port 8001
```

If the backend port changes, the frontend API configuration may also need to change.

### Model A fails to load

Check that both required resources are present and accessible:

- `backend/modelA/artifacts/modelA_rf_v6.pkl`
- `backend/models/pretrained/hand_landmarker.task`

Other possible causes include artifact corruption, configuration mismatches or dependency compatibility issues.

### Image prediction fails

- Check that the uploaded image is supported and decodable.
- Confirm that Model A is loaded.
- Confirm that the Hand Landmarker resource is available.
- Review the backend terminal for errors.
- Test the endpoint using `/docs`.

### Frontend cannot connect to the backend

- Confirm the backend is running.
- Check the backend URL configured in the frontend.
- Check the browser console and backend terminal for errors.
- Confirm the CORS origin is allowed.
- If the port or origin changes, update the relevant configuration.

Configured development origins:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

## 11. Project Evaluation & Verification

This section distinguishes tests performed during development from workflows that remain unverified.

### 11.1 Verified or tested during development

- Model A V6 training and evaluation
- Model artifact generation
- Model A inference pipeline
- Backend API operation
- API model-loading and image-prediction operations
- API documentation access
- Frontend production build

### 11.2 Model evaluation summary

| Parameter | Result |
|---|---|
| Model | Model A V6 |
| Algorithm | Random Forest |
| Feature version | V3 |
| Feature count | 95 |
| Classes | 29 |
| Train/test split | 80/20 |
| Test samples | 13,318 |
| Random state | 42 |
| Reported accuracy | 98.8287% |

The reported test accuracy applies to the documented evaluation dataset and split. It does not measure complete real-world continuous sign-language translation.

### 11.3 Verification status

| Functionality | Status |
|---|---|
| Model A V6 training | Tested |
| Model A test evaluation | Tested |
| Model artifact generation | Verified |
| Model A inference pipeline | Tested |
| Backend startup | Verified |
| API documentation access | Verified |
| Backend image prediction | Tested |
| Frontend production build | Successful |
| Full frontend-to-backend continuous webcam recognition | Not fully verified |
| Continuous sign-language sentence translation | Not verified as implemented |
| Model B integrated recognition | Not verified as integrated |
| Model C integrated recognition | Not verified as integrated |

Model evaluation, backend tests and frontend compilation do not independently prove that the complete application functions as a real-time continuous sign-language translation system under all conditions. Further end-to-end testing is needed.

## 12. Limitations

- **Static recognition scope:** Model A recognizes static hand gestures for the ASL alphabet and the `del`, `nothing` and `space` categories.
- **Not continuous translation:** The current implementation is not a complete continuous sign-language translation system.
- **Model conditions:** Prediction quality may be affected by lighting, background, camera quality, hand orientation, distance and individual signing styles.
- **Dataset limitations:** The dataset may not represent the full range of real-world signing conditions or linguistic diversity.
- **Temporal and linguistic context:** The current Model A pipeline does not establish recognition of temporal sequences, full grammar, sentence structure, facial expressions or complete body posture.
- **Webcam integration:** The complete continuous webcam experience through the frontend and backend remains unverified.
- **CPU processing:** Development and testing were conducted in a CPU-based environment without an NVIDIA GPU, which may affect processing speed.
- **Model B and Model C:** These are part of broader planned or experimental development and are not verified as integrated, production-ready systems.

The currently documented integrated recognition pipeline is Model A V6.

## 13. Credits, Attribution & License

### 13.1 Dataset attribution

HandLex uses an ASL Alphabet image dataset for training and evaluating static gesture recognition.

**The original dataset source, creator and license must be verified and credited here before public distribution.** Dataset reuse and redistribution must follow the original terms.

### 13.2 Libraries and frameworks

HandLex uses open-source tools and libraries, including:

- Python
- FastAPI
- Uvicorn
- React
- Vite
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- Pillow
- Python Multipart

XGBoost is also included in the dependency list, but is not identified as part of the active Model A inference pipeline.

These tools are subject to their respective licenses and usage terms.

### 13.3 Third-party resources and assets

Credit third-party models, resources, assets and libraries according to their respective licensing and attribution requirements. The MediaPipe Hand Landmarker resource should be acknowledged according to its distribution terms.

### 13.4 Project license

A project-specific license has not yet been finalized.

The project license should be selected after reviewing the code, dataset, trained model artifacts, pretrained resources and third-party dependencies.

## 14. Disclaimer

HandLex is an academic prototype developed for educational, experimental and demonstration purposes.

The system explores static sign gesture recognition using computer-vision and machine-learning techniques. Its predictions may be inaccurate or incomplete and should not be considered a reliable substitute for human interpretation of sign language.

HandLex is not a certified sign-language interpretation system and should not be relied upon for critical, medical, legal, emergency or other situations where accurate communication is essential.

The project demonstrates an exploration of machine learning for sign gesture recognition and is intended to support further experimentation and development.
