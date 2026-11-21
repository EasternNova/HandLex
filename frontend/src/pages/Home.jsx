import "../App.css";
import "../styles/Home.css";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <div className="logo">
          <span className="logo-mark">🤟</span>
          <span>HandLex</span>
        </div>

        <div className="nav-links">
          <Link to="/">Home</Link>
          <Link to="/translate">Translate</Link>
        </div>

        <Link to="/translate" className="nav-button">
          Get Started
        </Link>
      </nav>

      {/* Hero */}
      <main>
        <section className="hero-section" id="home">
          <div className="hero-content">
            <p className="eyebrow">AI-POWERED SIGN LANGUAGE</p>

            <h1>
              Breaking barriers.
              <br />
              <span>Building understanding.</span>
            </h1>

            <p className="hero-description">
              HandLex uses artificial intelligence to recognize sign language
              and help make communication more accessible.
            </p>

          <div className="hero-buttons">
            <Link to="/translate" className="primary-button">
              Start Translating →
            </Link>
           </div>
          </div>
          <div className="hero-visual">
            <div className="camera-card">
              <div className="camera-header">
                <span className="live-dot"></span>
                HandLex Vision
              </div>

              <div className="camera-placeholder">
                <div className="hand-icon">✋</div>
                <p>Camera recognition</p>
                <span>Ready to recognize signs</span>
              </div>

              <div className="prediction">
                <div>
                  <small>DETECTED SIGN</small>
                  <strong>Hello</strong>
                </div>

                <div className="confidence">
                  <small>CONFIDENCE</small>
                  <strong>--%</strong>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="how-section">
          <div className="section-heading">
            <p className="eyebrow">HOW IT WORKS</p>

            <h2>From movement to meaning.</h2>

            <p>
              Our AI pipeline turns hand movements into understandable text.
            </p>
          </div>

          <div className="steps">
            <div className="step">
              <span>01</span>

              <div className="step-icon">✋</div>

              <h3>Show a sign</h3>

              <p>
                Use your camera to show a sign in front of HandLex.
              </p>
            </div>

            <div className="step">
              <span>02</span>

              <div className="step-icon">👁️</div>

              <h3>AI detects</h3>

              <p>
                Computer vision identifies your hand and its landmarks.
              </p>
            </div>

            <div className="step">
              <span>03</span>

              <div className="step-icon">🧠</div>

              <h3>Model understands</h3>

              <p>
                Our machine-learning model analyzes the detected features.
              </p>
            </div>

            <div className="step">
              <span>04</span>

              <div className="step-icon">💬</div>

              <h3>Get your meaning</h3>

              <p>
                The recognized sign becomes text you can understand and use.
              </p>
            </div>
          </div>
        </section>

        {/* Vision */}
        <section className="vision-section">
          <div>
            <p className="eyebrow">THE BIGGER VISION</p>

            <h2>
              A bridge between
              <br />
              <span>sign and speech.</span>
            </h2>
          </div>

          <p className="vision-text">
            HandLex is being developed as an AI-powered platform for
            sign-language recognition, learning and communication. What starts
            with recognizing individual signs can grow into continuous
            sign-language understanding and two-way communication.
          </p>
        </section>
      </main>

      {/* Footer */}
      <footer>
        <div className="logo">
          <span className="logo-mark">🤟</span>
          <span>HandLex</span>
        </div>

        <p>AI-powered sign language learning & communication.</p>

        <span>© 2026 HandLex</span>
      </footer>
    </div>
  );
}

export default Home;