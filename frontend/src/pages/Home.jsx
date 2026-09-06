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
          <Link to="/learn">Learn</Link>
          <Link to="/practice">Practice</Link>
          <Link to="/dictionary">Dictionary</Link>
          <Link to="/about">About</Link>
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
              HandLex uses artificial intelligence to recognize sign language,
              help you learn, and make communication more accessible.
            </p>

            <div className="hero-buttons">
              <Link to="/translate" className="primary-button">
                Start Translating →
              </Link>

              <Link to="/learn" className="secondary-button">
                Explore Learning
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

        {/* Features */}
        <section className="features-section">
          <div className="section-heading">
            <p className="eyebrow">THE HANDLEX EXPERIENCE</p>

            <h2>More than recognition.</h2>
          </div>

          <div className="feature-grid">
            <div className="feature-card large">
              <div className="feature-icon">🤟</div>

              <h3>Sign → Text</h3>

              <p>
                Recognize signs through your camera and convert them into
                readable text.
              </p>

              <Link to="/translate">Try recognition →</Link>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📚</div>

              <h3>Learn</h3>

              <p>
                Learn the alphabet, words, phrases and everyday signs.
              </p>

              <Link to="/learn">Start learning →</Link>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🎯</div>

              <h3>Practice</h3>

              <p>
                Practice signs and receive instant feedback from the AI.
              </p>

              <Link to="/practice">Practice now →</Link>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📖</div>

              <h3>Sign Dictionary</h3>

              <p>
                Search and explore signs, meanings and demonstrations.
              </p>

              <Link to="/dictionary">Explore dictionary →</Link>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔊</div>

              <h3>Text → Speech</h3>

              <p>
                Turn recognized text into speech for easier communication.
              </p>

              <Link to="/translate">Learn more →</Link>
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