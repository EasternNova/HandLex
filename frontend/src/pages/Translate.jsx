
import { useEffect, useRef, useState } from "react";
import "../App.css";
import "../styles/Translate.css";

function Translate() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const requestRef = useRef(null);
  const busyRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [apiStatus, setApiStatus] = useState("Waiting");

  const API_URL = "http://127.0.0.1:8000/models/model-a/predict-image";

  const startCamera = async () => {
    try {
      setCameraError("");
      setPrediction(null);
      setApiStatus("Connecting");

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
    } catch (error) {
      console.error("Camera error:", error);
      setCameraActive(false);
      setApiStatus("Error");
      setCameraError(
        "Camera access was denied or is not available on this device."
      );
    }
  };

  const stopCamera = () => {
    if (requestRef.current) {
      requestRef.current.abort();
      requestRef.current = null;
    }

    busyRef.current = false;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setPrediction(null);
    setApiStatus("Waiting");
  };

  // Send camera frames to the backend
  useEffect(() => {
    if (!cameraActive) return;

    let intervalId;

    const sendFrame = async () => {
      const video = videoRef.current;

      if (!video || video.readyState < 2 || busyRef.current) return;

      busyRef.current = true;

      try {
        const canvas = document.createElement("canvas");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext("2d");

        if (!context) return;

        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise((resolve) =>
          canvas.toBlob(resolve, "image/jpeg", 0.8)
        );

        if (!blob) return;

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const controller = new AbortController();
        requestRef.current = controller;

        const response = await fetch(API_URL, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        setPrediction(data);

        if (data.committed) {
          setApiStatus("Recognized");
        } else if (data.hand_detected) {
          setApiStatus("Detecting");
        } else {
          setApiStatus("No hand");
        }
      } catch (error) {
        if (error.name !== "AbortError") {
          console.error("Prediction error:", error);
          setApiStatus("API Error");
        }
      } finally {
        requestRef.current = null;
        busyRef.current = false;
      }
    };

    intervalId = setInterval(sendFrame, 600);

    return () => {
      clearInterval(intervalId);

      if (requestRef.current) {
        requestRef.current.abort();
        requestRef.current = null;
      }

      busyRef.current = false;
    };
  }, [cameraActive]);

  // Cleanup when leaving the page
  useEffect(() => {
    return () => {
      if (requestRef.current) {
        requestRef.current.abort();
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const confidence = prediction?.confidence != null
    ? `${(prediction.confidence * 100).toFixed(1)}%`
    : "--%";

  const detectedSign = prediction?.hand_detected
    ? prediction.committed || prediction.prediction || "Detecting..."
    : cameraActive
      ? "No hand detected"
      : "Waiting...";

  const description = prediction?.committed
    ? "Sign recognized successfully."
    : prediction?.hand_detected
      ? "Hold your sign steady for recognition."
      : cameraActive
        ? "Position your hand clearly inside the camera frame."
        : "Show a sign in front of the camera to begin translation.";

  return (
    <div className="translate-page">

      {/* Header */}
      <div className="translate-header">
        <p className="eyebrow">AI SIGN TRANSLATOR</p>

        <h1>
          Sign <span>→</span> Text
        </h1>

        <p>
          Show a sign to your camera and let HandLex translate it into
          understandable text.
        </p>
      </div>

      {/* Translator */}
      <div className="translator-container">

        {/* CAMERA */}
        <div className="translator-camera">

          <div className="camera-top">
            <span className="live-dot"></span>

            HandLex Vision

            <span className="camera-status">
              {cameraActive ? "Camera active" : "Camera inactive"}
            </span>
          </div>

          <div className="camera-screen">

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`camera-video ${
                cameraActive ? "camera-video-active" : ""
              }`}
            />

            {!cameraActive && (
              <div className="translate-camera-placeholder">

                <div className="camera-icon">✋</div>

                <h3>Camera Ready</h3>

                <p>
                  Your camera feed will appear here when recognition starts.
                </p>

                <button
                  className="primary-button"
                  onClick={startCamera}
                >
                  Start Camera
                </button>

              </div>
            )}

            {cameraActive && (
              <button
                className="primary-button stop-camera-button"
                onClick={stopCamera}
              >
                Stop Camera
              </button>
            )}

            {cameraError && (
              <p className="camera-error">
                {cameraError}
              </p>
            )}

          </div>
        </div>

        {/* RESULT */}
        <div className="translation-result">

          <div className="result-header">
            <p className="eyebrow">TRANSLATION</p>

            <span>AI</span>
          </div>

          <div className="result-content">

            <small>DETECTED SIGN</small>

            <h2>{detectedSign}</h2>

            <p>{description}</p>

          </div>

          <div className="confidence-box">

            <div>
              <small>CONFIDENCE</small>

              <strong>{confidence}</strong>
            </div>

            <div>
              <small>STATUS</small>

              <strong>{apiStatus}</strong>
            </div>

          </div>

        </div>

      </div>

      {/* Instructions */}
      <section className="translate-instructions">

        <div>
          <span>01</span>

          <h3>Allow camera access</h3>

          <p>
            Give HandLex permission to access your camera.
          </p>
        </div>

        <div>
          <span>02</span>

          <h3>Show your sign</h3>

          <p>
            Position your hand clearly inside the camera frame.
          </p>
        </div>

        <div>
          <span>03</span>

          <h3>Get your translation</h3>

          <p>
            The AI model will recognize the sign and display the result.
          </p>
        </div>

      </section>

    </div>
  );
}

export default Translate;
