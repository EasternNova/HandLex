import { useEffect, useRef, useState } from "react";
import "../App.css";
import "../styles/Translate.css";

function Translate() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const startCamera = async () => {
    try {
      setCameraError("");

      // Prevent multiple camera streams
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
        },
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

      setCameraError(
        "Camera access was denied or is not available on this device."
      );
    }
  };

  const stopCamera = () => {
    // Stop every track
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    // Remove stream from video
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  };

  // Stop camera automatically when leaving the page
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });

        streamRef.current = null;
      }
    };
  }, []);

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

            {/* Video is ALWAYS present */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`camera-video ${
                cameraActive ? "camera-video-active" : ""
              }`}
            />


            {/* Camera not active */}
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


            {/* Camera active */}
            {cameraActive && (
              <button
                className="primary-button stop-camera-button"
                onClick={stopCamera}
              >
                Stop Camera
              </button>
            )}


            {/* Error */}
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

            <h2>Waiting...</h2>

            <p>
              Show a sign in front of the camera to begin translation.
            </p>

          </div>


          <div className="confidence-box">

            <div>
              <small>CONFIDENCE</small>

              <strong>--%</strong>
            </div>


            <div>
              <small>STATUS</small>

              <strong>
                {cameraActive ? "Camera Ready" : "Waiting"}
              </strong>
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