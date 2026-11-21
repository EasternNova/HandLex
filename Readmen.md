# 1. Project Overview

**HandLex** is a sign-language recognition and translation project designed to explore how computer vision and machine learning can help bridge the communication gap between sign-language users and people unfamiliar with sign language.

The project uses hand landmark detection and machine learning to recognize hand signs from visual input. By extracting hand-related features and processing them through a trained recognition model, HandLex aims to convert recognized signs into readable text through a web-based interface.

HandLex combines a Python-based machine learning backend with a React web frontend, providing an interface for interacting with the recognition system.

The project focuses on exploring sign recognition, feature extraction, model development, inference, and the integration of machine learning into a web application.

# 2. Project Objectives

## Primary Objectives

- Explore the application of computer vision and machine learning to sign-language hand-sign recognition.
- Recognize hand signs from visual input and convert recognized signs into corresponding text labels.
- Develop a machine-learning-based recognition system using extracted hand landmarks and related features.
- Provide a web-based interface for interacting with the sign recognition system.

## Intended Functionality

- Detect and extract hand landmarks from visual input using MediaPipe.
- Process extracted landmarks into numerical features.
- Use a trained machine learning model (Model A) to classify recognized hand signs.
- Display recognition results through the HandLex web interface.
- Provide a backend API for model loading, status monitoring, and image-based prediction.

## What the Project Aims to Demonstrate

- The use of computer vision for hand detection and landmark extraction.
- Feature engineering for machine-learning-based hand-sign recognition.
- Training, evaluation, and inference using a classification model.
- Integration of a machine-learning model with a backend API.
- Connecting a Python backend with a React-based frontend.
- The practical application of machine learning in an accessibility-oriented project.

# 3. Technology Stack

HandLex uses a combination of machine-learning libraries, computer-vision tools, backend technologies, and frontend technologies.

| Technology | Purpose | Usage in HandLex |
|---|---|---|
| **Python** | Programming language | Used for machine-learning development, feature extraction, model training, inference, and backend logic. |
| **FastAPI** | Backend API framework | Provides API endpoints for model management and image-based sign prediction. |
| **React** | Frontend library | Used to build the HandLex web interface and its interactive components. |
| **Vite** | Frontend development and build tool | Runs the frontend development server and builds the frontend for production. |
| **MediaPipe** | Computer-vision framework | Used for detecting and extracting hand landmarks from visual input. |
| **OpenCV** | Image and video processing library | Used for image processing, visual input handling, and camera-related operations. |
| **NumPy** | Numerical computing library | Used to process numerical data and construct feature arrays for model input. |
| **Scikit-learn** | Machine-learning library | Used for the Model A classification algorithm and related machine-learning operations. |
| **Joblib** | Serialization utility | Used to save and load the trained machine-learning model. |
| **Pillow (PIL)** | Image processing library | Supports image handling and processing within the Python environment. |
| **XGBoost** | Machine-learning library | Included in the project's dependency list for the broader machine-learning ecosystem and experiments. Its inclusion does not imply that it is used by the active Model A inference pipeline. |
| **Uvicorn** | ASGI server | Runs the FastAPI backend application. |
| **Python Multipart** | Form-data processing | Supports multipart form-data requests, including image uploads to the API. |

## Technology Categories

- **Programming language:** Python
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React, Vite
- **Computer vision:** MediaPipe, OpenCV, Pillow
- **Feature processing:** NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Model storage:** Joblib
- **API request handling:** Python Multipart

## 4. Repository Structure

HandLex is organized into separate directories for the machine-learning model, backend API, frontend interface, dataset, and project configuration.

```text
HandLex/
│
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   └── model_a_service.py
│   │
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
│   │
│   └── models/
│       └── pretrained/
│           └── hand_landmarker.task
│
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
│       │
│       └── asl_alphabet_test/
│           └── asl_alphabet_test/
│               ├── A_test.jpg
│               ├── B_test.jpg
│               ├── ...
│               └── Z_test.jpg
│
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
│
├── .gitignore
├── requirements.txt
├── README.md
└── ...
```

### Directory descriptions

| Directory / File | Purpose |
|---|---|
| `backend/api/` | Provides the FastAPI application and services that connect the machine-learning model to the frontend. |
| `backend/modelA/` | Contains Model A's feature extraction, inference, training, landmark processing and prediction-related scripts. |
| `backend/modelA/artifacts/` | Stores the trained model artifact and its associated metadata. |
| `backend/modelA/checkpoints/` | Stores model training and experimental checkpoints. |
| `backend/modelA/data/` | Contains prepared datasets and processed data used by Model A experiments. |
| `backend/models/pretrained/` | Stores pretrained resources required for hand landmark detection. |
| `dataset/ASL/` | Contains the ASL Alphabet training and testing images. |
| `frontend/src/` | Contains the React application source code, pages and styles. |
| `frontend/public/` | Stores static frontend assets. |
| `frontend/package.json` | Defines frontend dependencies and development/build scripts. |
| `requirements.txt` | Lists Python dependencies for the backend and machine-learning environment. |

The repository tree excludes generated files and directories such as `node_modules`, `dist`, `__pycache__`, virtual environments and other temporary or build-related files.

## 5. Dataset & Data Preparation

### 5.1 Dataset

HandLex Model A uses an ASL Alphabet image dataset containing images of hand gestures representing letters of the American Sign Language (ASL) alphabet and additional gesture categories.

The dataset is used to train and evaluate a machine-learning model for static sign recognition.

### 5.2 Dataset Structure

The dataset is organized into training and testing directories.

```text
dataset/
└── ASL/
    ├── asl_alphabet_train/
    │   └── asl_alphabet_train/
    │       ├── A/
    │       ├── B/
    │       ├── C/
    │       ├── ...
    │       ├── Z/
    │       ├── del/
    │       ├── nothing/
    │       └── space/
    └── asl_alphabet_test/
        └── asl_alphabet_test/
            ├── A_test.jpg
            ├── B_test.jpg
            ├── ...
            ├── Z_test.jpg
            ├── del_test.jpg
            ├── nothing_test.jpg
            └── space_test.jpg
```

### 5.3 Classes and Labels

Model A recognizes 29 classes:

- **26 alphabet classes:** A–Z
- **3 additional classes:** `del`, `nothing`, and `space`

These labels represent the classification categories used by the model.

### 5.4 Dataset Attribution

The dataset is based on an ASL Alphabet image dataset. The original source and dataset creator should be properly credited in the final documentation.

> The exact dataset source and original attribution should be confirmed before the README is finalized.

### 5.5 License and Usage Restrictions

The dataset's original license and usage conditions must be respected. Dataset redistribution and reuse may be subject to restrictions imposed by the original creator.

The dataset license should be verified from its original source before redistribution.

### 5.6 Data Preprocessing

The images are processed to extract hand-related information rather than relying directly on the raw image pixels for Model A's classification pipeline.

The preprocessing pipeline involves:

1. Loading the input image.
2. Detecting hand landmarks using MediaPipe Hand Landmarker.
3. Extracting hand landmark coordinates.
4. Processing landmark information into numerical features.
5. Preparing the extracted features for machine-learning classification.

### 5.7 Train/Test Splitting

For the reported Model A V6 evaluation, the data was divided using a stratified 80/20 train/test split.

- **Training set:** 80%
- **Testing set:** 20%
- **Random state:** 42
- **Stratification:** Used to maintain class distribution across the split.

The reported evaluation was conducted on 13,318 test samples.

### 5.8 Feature Preparation

Model A uses a feature-engineering pipeline to convert detected hand landmarks into numerical representations.

The V3 feature pipeline generates **95 features** consisting of:

- **63 landmark coordinate features:** Normalized XYZ coordinates of hand landmarks.
- **10 joint-angle features:** Representations of finger and joint geometry.
- **20 wrist-distance features:** Distance-based measurements relative to the wrist.
- **Additional features:** Hand presence and handedness-related information.

These numerical features are supplied to the Random Forest classifier for sign classification.

## 6. Frontend — HandLex Web Interface

### 6.1 Frontend Architecture

HandLex uses a React-based frontend to provide the user interface for interacting with the sign recognition system.

The frontend is responsible for displaying project information, providing the translation interface, handling user interactions, and communicating with the backend API.

### 6.2 React and Vite

- **React:** Used to build the frontend interface using reusable components and structured UI logic.
- **Vite:** Used as the frontend development and build tool, providing a development server and production build process.

### 6.3 Pages

The frontend is organized into pages that provide different parts of the HandLex experience.

### 6.4 Components

React components are used to organize the interface into manageable and reusable sections, helping separate UI elements from the overall application structure.

### 6.5 Home Page

The Home page serves as the main entry point to the HandLex website.

It introduces the project, presents its purpose, and provides navigation to the main features of the application.

### 6.6 Translate Page

The Translate page provides the interface for interacting with the sign recognition functionality.

It is designed to connect the frontend interface with the backend recognition system and display the results of recognition requests.

### 6.7 Backend Communication

The frontend communicates with the FastAPI backend through HTTP API requests.

The backend handles model-related operations and image prediction requests, while the frontend handles the interface and displays the returned information.

### 6.8 Frontend Development Server

The frontend is developed using Vite.

To start the frontend development server:

```bash
cd frontend
npm install
npm run dev
```

By default, Vite runs the development server at:

`http://localhost:5173`

The development server allows the frontend to be accessed locally during development and testing.

## 7. Backend and Machine Learning

### 7.A — FastAPI Backend

#### 7.A.1 Backend Architecture

HandLex uses FastAPI as its backend framework. The backend connects the frontend interface to the sign-recognition system and manages model operations and image prediction requests.

The backend separates API routing and request handling from the model inference logic. This allows the API to interact with Model A through a dedicated model service.

#### 7.A.2 Backend Components

The backend is organized into components responsible for API operations and model inference.

| Component | File | Responsibility |
|---|---|---|
| API application | `backend/api/main.py` | Defines API routes, health checks, model operations and image prediction requests. |
| Model service | `backend/api/model_a_service.py` | Manages Model A operations and connects the API to the inference engine. |
| Inference engine | `backend/modelA/inference.py` | Handles the recognition pipeline, including feature preparation and classification. |
| Feature extraction | `backend/modelA/features.py` | Converts hand landmark information into numerical features. |
| Hand landmark detection | `backend/modelA/landmarker.py` | Handles hand landmark detection using MediaPipe. |
| Prediction stabilization | `backend/modelA/prediction_stabilizer.py` | Helps stabilize repeated prediction results. |

#### 7.A.3 Model Loading

Model A is managed through the backend model service.

The backend supports explicit model loading and unloading operations. This allows the model to be initialized when required and released when it is no longer needed.

The model-loading process involves checking the required model resources, loading the trained classifier and preparing the inference pipeline.

#### 7.A.4 Inference Process

The backend inference process follows a sequence of operations to convert an uploaded image into a classification result.

1. The frontend sends an image to the backend.
2. FastAPI receives the uploaded image.
3. The image is decoded into an image frame.
4. MediaPipe Hand Landmarker detects the hand and extracts landmarks.
5. The landmark information is processed into a 95-feature vector.
6. The feature vector is supplied to the trained Model A classifier.
7. The classifier generates a prediction.
8. The prediction result is returned through the API.

#### 7.A.5 API Endpoints

The backend exposes endpoints for API status, health checking, model management and image prediction.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Provides API welcome or status information. |
| `GET` | `/health` | Checks backend health. |
| `GET` | `/models/model-a/status` | Checks Model A status and resource availability. |
| `POST` | `/models/model-a/load` | Loads Model A. |
| `POST` | `/models/model-a/unload` | Unloads Model A. |
| `POST` | `/models/model-a/predict-image` | Accepts an image and returns a prediction result. |

Detailed endpoint documentation is provided in Section 11 — API Reference.

#### 7.A.6 Request and Response Handling

The backend receives requests from the frontend through HTTP endpoints.

The image prediction endpoint accepts an uploaded image using multipart/form-data. The image is processed by the inference pipeline and the prediction result is returned as a JSON response.

Other endpoints return JSON information related to API status, health, model availability or model operations.

#### 7.A.7 API Documentation

FastAPI automatically generates interactive API documentation using the OpenAPI specification.

The documentation is available locally at:

`http://localhost:8000/docs`

The interactive documentation allows developers to:

- View available endpoints.
- Inspect endpoint parameters and request requirements.
- Test API operations.
- Submit image prediction requests.
- Inspect API responses.

#### 7.A.8 CORS Configuration

Cross-Origin Resource Sharing (CORS) is configured to allow the HandLex frontend development server to communicate with the backend.

The currently configured frontend origins are:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

The configuration allows credentials, HTTP methods and headers for these origins.

---

### 7.B — Model A: Sign Recognition

#### 7.B.1 Model Overview

Model A is HandLex's static sign classification model, developed to recognize hand gestures corresponding to the letters of the ASL alphabet and additional gesture categories.

The model uses hand landmark information extracted from images and converts it into numerical features for classification.

Instead of classifying raw image pixels directly, Model A uses geometric representations of the detected hand. These representations are processed by a machine-learning classifier to generate a predicted gesture class.

Model A V6 uses a Random Forest classifier and a Feature Version 3 (V3) feature extraction pipeline.

#### 7.B.2 Recognition Approach

Model A follows a landmark-based machine-learning approach.

The recognition pipeline combines hand landmark detection, feature engineering and supervised classification.

The general workflow is:

1. Receive an image containing a hand gesture.
2. Detect the hand using MediaPipe Hand Landmarker.
3. Extract the 21 hand landmarks.
4. Transform the landmark information into a structured numerical feature vector.
5. Pass the feature vector to the trained classifier.
6. Obtain the predicted gesture class.
7. Apply prediction stabilization where configured.

This approach converts the visual geometry of a hand into numerical data that can be processed by a traditional machine-learning algorithm.

#### 7.B.3 Model Architecture and Algorithm

Model A V6 uses a **Random Forest Classifier**.

Random Forest is an ensemble-learning algorithm that combines multiple decision trees to produce a classification result.

During training, the classifier learns patterns from labeled feature data. During prediction, the trained model processes a new feature vector and determines the class based on the learned decision-tree ensemble.

The Random Forest classifier is used to classify the extracted hand features into one of the model's 29 trained classes.

#### 7.B.4 Feature Extraction

Model A uses hand landmarks detected by MediaPipe Hand Landmarker.

MediaPipe provides 21 landmarks representing important points of the hand, including the wrist, finger joints and fingertips.

Each landmark contains X, Y and Z coordinate information. These coordinates are processed and transformed into geometric features representing the hand's structure and spatial relationships.

The feature extraction process combines normalized landmark coordinates, joint-angle measurements and wrist-distance measurements, along with additional hand-related indicators.

The resulting numerical feature vector is supplied to the Random Forest classifier.

#### 7.B.5 Feature Version V3 — 95 Features

Model A uses Feature Version V3, which generates a total of 95 numerical features.

| Feature group | Count | Description |
|---|---:|---|
| Normalized XYZ coordinates | 63 | XYZ coordinate values for the 21 hand landmarks. |
| Joint-angle features | 10 | Geometric angle representations of finger and joint relationships. |
| Wrist-distance features | 20 | Distance measurements relative to the wrist reference. |
| Hand-presence indicator | 1 | Hand-presence-related information. |
| Handedness indicator | 1 | Handedness-related information. |
| **Total** | **95** | **Complete feature vector.** |

These features provide the classifier with a numerical representation of the hand's geometry.

#### 7.B.6 Classification Classes

Model A is trained to recognize 29 classification categories.

**Alphabet classes:**

- A
- B
- C
- D
- E
- F
- G
- H
- I
- J
- K
- L
- M
- N
- O
- P
- Q
- R
- S
- T
- U
- V
- W
- X
- Y
- Z

**Additional classes:**

| Class | Description |
|---|---|
| `del` | Deletion-related gesture category. |
| `nothing` | Category representing no active sign. |
| `space` | Space-related gesture category. |

The classifier's output corresponds to one of these 29 trained labels.

#### 7.B.7 Training Process

Model A follows a supervised machine-learning training process.

The training workflow involves:

1. Preparing the dataset.
2. Preparing the numerical feature representations.
3. Associating feature vectors with their corresponding class labels.
4. Splitting the dataset into training and testing subsets.
5. Training the Random Forest classifier using the training data.
6. Evaluating the trained classifier.
7. Saving the trained model artifact for later inference.

The trained model is saved as an artifact so it can be loaded independently during backend inference without retraining.

#### 7.B.8 Model Artifact

The trained Model A V6 artifact is stored at:

`backend/modelA/artifacts/modelA_rf_v6.pkl`

The artifact contains the trained classifier required for classification.

Associated model metadata is used by the inference pipeline to identify the expected feature configuration and classification information.

The model artifact is loaded by the backend model service when Model A is initialized.

#### 7.B.9 Evaluation Methodology

Model A V6 was evaluated using a stratified train/test split.

| Evaluation parameter | Value |
|---|---|
| Training proportion | 80% |
| Testing proportion | 20% |
| Splitting method | Stratified |
| Random state | 42 |
| Test samples | 13,318 |

Stratified splitting was used to maintain class representation across the training and testing subsets.

The test dataset was used to evaluate the classification performance of the trained model.

#### 7.B.10 Evaluation Results

**Reported test accuracy: 98.8287%**

Model A V6 achieved a reported accuracy of 98.8287% on the evaluated test dataset.

Accuracy represents the proportion of test samples that were correctly classified by the model.

This result reflects the model's performance under the documented evaluation setup. It should not be interpreted as a guarantee of accuracy under every real-world condition, including varying lighting, backgrounds, camera quality, hand orientation, distance or individual signing styles.

#### 7.B.11 Prediction Stabilization

Model A includes a prediction stabilization component intended to reduce unstable predictions across repeated inference results.

The stabilization configuration includes:

| Parameter | Value |
|---|---:|
| History size | 7 |
| Stable prediction count | 5 |
| Confidence threshold | 0.50 |

The stabilization mechanism processes repeated prediction information to help reduce prediction fluctuations.

It is intended to improve consistency in repeated recognition outputs. Its presence in the inference pipeline does not imply that a fully continuous sign-language translation system is already implemented.

#### 7.B.12 Model Limitations

Model A currently focuses on static gesture classification.

Its recognition capabilities depend on the characteristics of its training data, feature representation and input conditions.

Static hand landmark classification does not inherently capture:

- Temporal relationships between consecutive signs.
- Full sign-language grammar.
- Sentence structure.
- Facial expressions.
- Body movements and posture.
- Context-dependent meaning.

Model A should therefore be considered a static gesture recognition component of the broader HandLex project rather than a complete sign-language translation system.

---

### 7.C — Pretrained Model Resources

#### 7.C.1 MediaPipe Hand Landmarker

HandLex uses the pretrained MediaPipe Hand Landmarker model for hand detection and landmark extraction.

The pretrained model file is located at:

`backend/models/pretrained/hand_landmarker.task`

#### 7.C.2 Purpose

The MediaPipe Hand Landmarker is responsible for detecting hand landmarks from an image.

It provides the landmark information used by the Model A feature extraction pipeline.

The detected landmarks are transformed into numerical features that are subsequently supplied to the Random Forest classifier.

#### 7.C.3 Model Dependency

The pretrained Hand Landmarker file is a required resource for the Model A inference pipeline.

The inference engine verifies the availability of the required landmark model resource before initializing the recognition pipeline.

The pretrained landmark model and the trained Model A classifier serve different purposes:

| Model/resource | Purpose |
|---|---|
| MediaPipe Hand Landmarker | Detects and extracts hand landmarks from an image. |
| HandLex Model A V6 | Classifies the extracted hand features into one of 29 trained classes. |

Both components are required for the documented Model A image-recognition pipeline.

## 8. Installation & Running HandLex

### 8.1 Prerequisites

Before running HandLex, ensure the following software is installed:

| Software | Purpose |
|---|---|
| Python | Backend development and machine-learning execution. |
| Node.js | Required to run the frontend development environment. |
| npm | Frontend dependency management. |
| Git | Repository management (optional if the repository is downloaded directly). |

### 8.2 Python Environment

HandLex uses a Python virtual environment to isolate backend dependencies from other Python installations.

From the project root:

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is successful, the terminal will display the virtual environment name.

### 8.3 Dependency Installation

Install the Python dependencies from the project root:

```powershell
pip install -r requirements.txt
```

The frontend dependencies are installed separately from the frontend directory.

### 8.4 Backend Startup

From the project root:

```powershell
cd backend
uvicorn api.main:app --reload
```

The FastAPI backend starts using Uvicorn, with automatic reload enabled during development.

### 8.5 Frontend Startup

Open a second terminal from the project root.

Navigate to the frontend directory:

```powershell
cd frontend
```

Install the frontend dependencies:

```powershell
npm install
```

Start the Vite development server:

```powershell
npm run dev
```

### 8.6 Local URLs

Once the backend and frontend are running, the following local addresses are available:

| Service | URL |
|---|---|
| HandLex Frontend | `http://localhost:5173` |
| FastAPI Backend | `http://localhost:8000` |
| API Documentation | `http://localhost:8000/docs` |
| API Root | `http://localhost:8000/` |

### 8.7 First-Time Setup

For a first-time local setup:

1. Clone or download the HandLex repository.
2. Install Python and Node.js.
3. Create and activate the Python virtual environment.
4. Install Python dependencies.
5. Install frontend dependencies.
6. Verify that the required pretrained Hand Landmarker file is available.
7. Verify that the trained Model A artifact is available.
8. Start the backend.
9. Start the frontend.
10. Open the HandLex website in a browser.
11. Check the API documentation if backend verification is required.

Required model resources:

- `backend/models/pretrained/hand_landmarker.task`
- `backend/modelA/artifacts/modelA_rf_v6.pkl`

### 8.8 Shutdown Instructions

To stop the backend or frontend development server:

1. Navigate to the terminal running the server.
2. Press `Ctrl + C`.
3. Wait for the process to terminate.

To deactivate the Python virtual environment:

```powershell
deactivate
```

---

## 9. Troubleshooting

### 9.1 Python Environment Issues

**Problem:** Python is not recognized by the terminal.

**Possible solution:**

- Verify that Python is installed.
- Check whether Python is added to the system PATH.
- Restart the terminal after changing PATH settings.
- Verify the installation:

```powershell
python --version
```

If Python is unavailable through `python`, try:

```powershell
py --version
```

**Problem:** Virtual environment activation fails in PowerShell.

PowerShell may restrict script execution.

If activation is blocked, use an appropriate PowerShell execution policy for your environment or use another terminal such as Command Prompt.

### 9.2 Package Installation Problems

**Problem:** Dependencies fail to install.

**Possible solutions:**

- Verify the Python version.
- Confirm that the virtual environment is activated.
- Ensure that pip is available.
- Check the internet connection.
- Check whether the required package supports the installed Python version.

Verify pip:

```powershell
python -m pip --version
```

### 9.3 Backend Startup Errors

**Problem:** The FastAPI server fails to start.

**Possible solutions:**

- Verify that the virtual environment is activated.
- Confirm that the backend dependencies are installed.
- Run the command from the correct directory.
- Check the terminal for import errors.
- Verify the API entry point.

Expected backend startup command from the project root:

```powershell
cd backend
uvicorn api.main:app --reload
```

### 9.4 Port Conflicts

**Problem:** The server cannot start because the port is already in use.

**Possible solutions:**

- Check whether another process is using the port.
- Stop the existing server if it is no longer required.
- Configure the server to use another available port.

For example, to run the backend on another port:

```powershell
uvicorn api.main:app --reload --port 8001
```

If the backend port changes, the frontend API configuration may also need to be updated.

### 9.5 Model Loading Errors

**Problem:** Model A cannot be loaded.

**Possible causes:**

- The model artifact is missing.
- The pretrained Hand Landmarker file is missing.
- The artifact is corrupted.
- The model configuration does not match the expected feature configuration.
- A dependency compatibility issue exists.

**Required resources:**

```text
backend/
├── modelA/
│   └── artifacts/
│       └── modelA_rf_v6.pkl
└── models/
    └── pretrained/
        └── hand_landmarker.task
```

Verify that both files exist at their expected locations.

### 9.6 Image Prediction Errors

**Problem:** Image prediction fails.

**Possible solutions:**

- Verify that the uploaded file is a supported image.
- Ensure the image can be decoded.
- Confirm that Model A is loaded.
- Verify that the Hand Landmarker resource is available.
- Check the backend terminal for error messages.
- Test the endpoint through the FastAPI documentation.

### 9.7 Webcam Issues

**Problem:** Webcam access or capture does not work.

**Possible solutions:**

- Verify that the camera is connected.
- Check operating-system camera permissions.
- Ensure that another application is not using the camera.
- Verify that the correct camera device is selected.
- Check the webcam script and its configuration.
- Test the camera independently if necessary.

Webcam-related functionality may depend on the specific implementation and environment.

### 9.8 Frontend/API Connection Problems

**Problem:** The frontend cannot communicate with the backend.

**Possible solutions:**

- Confirm that the backend server is running.
- Verify that the backend URL configured in the frontend is correct.
- Check the browser console for network errors.
- Check the backend terminal for API errors.
- Verify the CORS configuration.
- Ensure that the frontend origin is permitted by the backend.

The configured development origins are:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

If either the frontend or backend is running on a different port or origin, the configuration may need to be updated.

## 10. API Reference

HandLex uses a FastAPI backend to provide health checks, model management and image-based sign recognition.

The API endpoints allow the frontend and other clients to interact with the backend and Model A.

### 10.1 Endpoint Overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Returns API information or status. |
| `GET` | `/health` | Checks backend health. |
| `GET` | `/models/model-a/status` | Retrieves Model A status information. |
| `POST` | `/models/model-a/load` | Loads Model A into the backend. |
| `POST` | `/models/model-a/unload` | Unloads Model A from the backend. |
| `POST` | `/models/model-a/predict-image` | Accepts an image and returns a prediction result. |

---

### 10.2 `GET /`

**Purpose:** Returns information about the HandLex API.

| Property | Description |
|---|---|
| Method | `GET` |
| URL | `/` |
| Input | No request body required. |
| Output | JSON response containing API information. |

**Example request:**

```http
GET http://localhost:8000/
```

**Example usage:**

Open the endpoint in a browser to check the API root response.

---

### 10.3 `GET /health`

**Purpose:** Checks the health of the backend service.

| Property | Description |
|---|---|
| Method | `GET` |
| URL | `/health` |
| Input | No request body required. |
| Output | JSON health response. |

**Example request:**

```http
GET http://localhost:8000/health
```

This endpoint can be used to check whether the backend health-check route is responding.

---

### 10.4 `GET /models/model-a/status`

**Purpose:** Retrieves information about the availability and status of Model A.

| Property | Description |
|---|---|
| Method | `GET` |
| URL | `/models/model-a/status` |
| Input | No request body required. |
| Output | JSON response containing model status information. |

**Example request:**

```http
GET http://localhost:8000/models/model-a/status
```

This endpoint can be used to inspect the model's status before performing image prediction.

---

### 10.5 `POST /models/model-a/load`

**Purpose:** Loads Model A and prepares the model for inference.

| Property | Description |
|---|---|
| Method | `POST` |
| URL | `/models/model-a/load` |
| Input | No image or request body required, according to the documented endpoint operation. |
| Output | JSON response indicating the result of the loading operation. |

**Example request:**

```http
POST http://localhost:8000/models/model-a/load
```

**Example usage:**

Use this endpoint when Model A needs to be initialized before prediction requests.

---

### 10.6 `POST /models/model-a/unload`

**Purpose:** Unloads Model A from the backend.

| Property | Description |
|---|---|
| Method | `POST` |
| URL | `/models/model-a/unload` |
| Input | No image or request body required, according to the documented endpoint operation. |
| Output | JSON response indicating the result of the unloading operation. |

**Example request:**

```http
POST http://localhost:8000/models/model-a/unload
```

**Example usage:**

Use this endpoint to release the loaded Model A resources when the model is no longer required.

---

### 10.7 `POST /models/model-a/predict-image`

**Purpose:** Accepts an image containing a hand gesture and processes it using Model A.

| Property | Description |
|---|---|
| Method | `POST` |
| URL | `/models/model-a/predict-image` |
| Input | Uploaded image using multipart/form-data. |
| Output | JSON response containing the prediction result and associated information. |

#### Request

The endpoint accepts an image file through a multipart/form-data request.

The uploaded image is processed through the Model A inference pipeline.

#### Example request

```http
POST http://localhost:8000/models/model-a/predict-image
Content-Type: multipart/form-data
```

The request must include the image file as the upload field expected by the API.

#### Example using cURL

```bash
curl -X POST "http://localhost:8000/models/model-a/predict-image" -F "file=@hand.jpg"
```

<text color="warning" weight="medium">Note:</text> The upload field name shown as `file` in this example should be verified against the actual FastAPI endpoint declaration before finalizing the README.

#### Output

The endpoint returns a JSON response containing the classification result and prediction-related information.

The exact response structure depends on the implementation of the prediction endpoint.

---

### 10.8 Interactive API Testing

HandLex provides interactive API documentation through FastAPI.

**URL:**

`http://localhost:8000/docs`

The Swagger interface can be used to:

- Inspect available API endpoints.
- View request parameters and expected input formats.
- Execute API requests.
- Upload images for prediction.
- Inspect API responses.

The interactive documentation is particularly useful for testing the backend independently of the frontend.

## 11. Limitations

HandLex is an ongoing sign-language recognition project. Although Model A has been trained and integrated into the backend inference pipeline, the current implementation has limitations that affect its scope, performance and usability.

### 11.1 Current Recognition Scope

The current Model A implementation focuses on static gesture classification.

It recognizes individual hand gestures corresponding to the ASL alphabet and three additional categories: `del`, `nothing` and `space`.

It is not currently a complete continuous sign-language translation system capable of translating full conversations.

### 11.2 Model Limitations

Model A uses landmark-based feature extraction and a Random Forest classifier.

Its performance depends on successful hand landmark detection and the quality of the extracted features. Variations in hand position, orientation, lighting, background, camera quality and signing style may affect prediction results.

The reported evaluation accuracy should not be interpreted as guaranteed real-world accuracy.

### 11.3 Dataset Limitations

The model's capabilities are influenced by the dataset used for training and evaluation.

The training dataset may not represent the full diversity of real-world signing conditions, including differences in hand appearance, backgrounds, lighting, camera angles and individual signing styles.

The current dataset and classification scope do not cover the complete linguistic structure of sign language.

### 11.4 Webcam Limitations

The webcam-related functionality is not fully verified as an integrated end-to-end recognition experience.

Although webcam-related scripts and camera testing exist within the project, reliable continuous webcam recognition through the complete HandLex frontend and backend workflow remains unverified.

### 11.5 CPU-Based Processing

HandLex was developed and tested in a CPU-based environment without an NVIDIA GPU.

The current processing environment uses CPU-based execution, which may affect inference speed and processing performance compared with systems equipped with supported GPU acceleration.

### 11.6 Features Not Currently Integrated

The current implementation does not establish a complete continuous sign-language translation system.

Features such as full sentence translation, temporal sequence recognition, sign-language grammar processing, facial expression interpretation and complete conversational translation are not currently established as integrated capabilities.

### 11.7 Model B and Model C Status

Model B and Model C are part of the broader planned or experimental model development work within HandLex.

Their associated development materials and experimental work should not be interpreted as fully integrated, production-ready recognition systems.

The currently documented integrated recognition pipeline is Model A V6.

---

## 12. Project Evaluation & Verification

HandLex was evaluated through model testing, backend verification and frontend build checks. The following section distinguishes completed verification from functionality that remains unverified.

### 12.1 What Has Been Tested

The following areas have been tested or verified during development:

- Model A V6 training and evaluation.
- Model artifact generation.
- Model A inference functionality.
- Backend API operation.
- API model loading and image prediction.
- Backend API documentation access.
- Frontend build process.

### 12.2 Model Evaluation Results

Model A V6 was evaluated using the documented stratified train/test split.

| Parameter | Value |
|---|---|
| Model | Model A V6 |
| Algorithm | Random Forest Classifier |
| Feature version | V3 |
| Feature count | 95 |
| Classes | 29 |
| Training split | 80% |
| Testing split | 20% |
| Test samples | 13,318 |
| Random state | 42 |
| Reported accuracy | 98.8287% |

The reported test accuracy represents the model's performance on the documented evaluation dataset and split.

It is not a measurement of full real-world continuous sign-language translation performance.

### 12.3 API Verification

The backend API was tested during development.

The following areas were verified:

- Backend API startup.
- API documentation access through `/docs`.
- Model loading and inference-related backend operations.
- Image prediction request processing.
- Successful API responses during testing.

The API was tested independently of a fully integrated continuous webcam translation workflow.

### 12.4 Frontend Build Verification

The React frontend was successfully built using Vite during development.

The production build completed successfully, confirming that the frontend source code could be compiled into a build output.

Build verification does not necessarily establish that every frontend feature, API integration or real-time recognition workflow has been fully tested.

### 12.5 Verified vs. Unverified Functionality

| Functionality | Verification status |
|---|---|
| Model A V6 training | Tested |
| Model A V6 test evaluation | Tested |
| Model A artifact generation | Verified |
| Model A inference pipeline | Tested |
| Backend startup | Verified |
| API documentation access | Verified |
| Backend image prediction | Tested |
| Frontend production build | Successful |
| Full frontend-to-backend continuous webcam recognition | Not fully verified |
| Continuous sign-language sentence translation | Not implemented as a verified integrated capability |
| Model B integrated recognition | Not verified as an integrated system |
| Model C integrated recognition | Not verified as an integrated system |

### 12.6 Verification Scope

The verification performed during development establishes that the Model A classification pipeline, backend API operations and frontend build process have been tested in their respective environments.

However, successful model evaluation, backend testing and frontend compilation do not independently establish that the complete HandLex application functions as a real-time, continuous sign-language translation system under all operating conditions.

Further integrated testing is required to validate the complete user-facing recognition workflow.

## 13. Credits, Attribution & License

### 13.1 Dataset Attribution

HandLex uses an ASL Alphabet image dataset for training and evaluating its static gesture recognition model.

The original dataset creator, source and license should be acknowledged according to the dataset's original terms.

<text>Dataset source and license details will be added after the original dataset source has been verified.</text>

### 13.2 Libraries and Frameworks

HandLex is built using open-source libraries, frameworks and tools, including:

| Technology | Purpose |
|---|---|
| Python | Backend and machine-learning development. |
| FastAPI | Backend API framework. |
| React | Frontend user interface. |
| Vite | Frontend development and build tool. |
| MediaPipe | Hand landmark detection. |
| OpenCV | Image processing. |
| NumPy | Numerical processing. |
| Scikit-learn | Machine-learning model development and classification. |
| Joblib | Model serialization and loading. |
| Pillow | Image processing support. |
| Uvicorn | ASGI server for running the backend. |

These technologies are used in accordance with their respective licenses and usage terms.

### 13.3 Third-Party Assets

Third-party assets, if used, should be credited according to their respective licenses and attribution requirements.

### 13.4 Project License

A project-specific license has not yet been finalized.

The licensing of HandLex should be determined after reviewing the project's code, dataset, model artifacts and third-party dependencies to ensure that the selected license is appropriate.

---

## 14. Disclaimer

HandLex is an academic prototype developed for educational, experimental and demonstration purposes.

The system is designed to explore static sign gesture recognition using machine-learning techniques. Its predictions may be inaccurate or incomplete and should not be considered a reliable substitute for human interpretation of sign language.

HandLex is not a certified sign-language interpretation system and should not be relied upon for critical, medical, legal, emergency or other situations where accurate communication is essential.

The project is intended to demonstrate the application of computer vision and machine learning to sign gesture recognition and to support further experimentation and development.