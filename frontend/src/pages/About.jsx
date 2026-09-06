import { Link } from "react-router-dom";

import "../App.css";
import "../styles/About.css";

function About() {
  return (
    <main className="about-page">
      {/* Header */}
      <section className="about-header">
        <p className="eyebrow">ABOUT HANDLEX</p>

        <h1>
          Technology that helps people <span>communicate.</span>
        </h1>

        <p>
          HandLex is an AI-powered sign language platform designed
          to make sign language learning and communication more
          accessible through technology.
        </p>
      </section>

      {/* Mission */}
      <section className="about-mission">
        <div className="about-mission-label">
          <p className="eyebrow">OUR MISSION</p>
        </div>

        <div className="about-mission-content">
          <h2>
            Breaking barriers.
            <br />
            Building <span>understanding.</span>
          </h2>

          <p>
            Communication should not be limited by the way we
            speak. HandLex explores how artificial intelligence,
            computer vision, and sign language technology can work
            together to create more inclusive communication.
          </p>

          <p>
            Our goal is to build technology that can understand
            sign language, help people learn it, and eventually
            translate between signed and spoken communication.
          </p>
        </div>
      </section>

      {/* How it works */}
      <section className="about-technology">
        <div className="section-heading">
          <p className="eyebrow">THE TECHNOLOGY</p>

          <h2>
            Built around <span>AI.</span>
          </h2>

          <p>
            HandLex combines computer vision, machine learning,
            and modern web technology into one platform.
          </p>
        </div>

        <div className="about-tech-grid">
          <article className="about-tech-card">
            <div className="about-tech-number">01</div>

            <div className="about-tech-icon">👁️</div>

            <h3>Computer Vision</h3>

            <p>
              Hand and body landmarks can be extracted from
              camera input to represent signing as structured
              data.
            </p>
          </article>

          <article className="about-tech-card">
            <div className="about-tech-number">02</div>

            <div className="about-tech-icon">🧠</div>

            <h3>Machine Learning</h3>

            <p>
              Machine learning models analyze movement and
              landmark patterns to recognize signs.
            </p>
          </article>

          <article className="about-tech-card">
            <div className="about-tech-number">03</div>

            <div className="about-tech-icon">⚡</div>

            <h3>Real-Time AI</h3>

            <p>
              The platform is designed to connect recognition
              models with a real-time web interface.
            </p>
          </article>

          <article className="about-tech-card">
            <div className="about-tech-number">04</div>

            <div className="about-tech-icon">🌐</div>

            <h3>Accessible Web Platform</h3>

            <p>
              A modern interface brings learning, practice,
              recognition, and communication tools together.
            </p>
          </article>
        </div>
      </section>

      {/* Vision */}
      <section className="about-vision">
        <div className="about-vision-content">
          <p className="eyebrow">THE BIGGER VISION</p>

          <h2>
            From recognizing signs to enabling
            <span> communication.</span>
          </h2>

          <p>
            HandLex starts with sign recognition, but the larger
            vision goes further. Future versions can explore
            continuous sign language understanding, text-to-sign
            communication, personalized learning, and AI-powered
            signing assistance.
          </p>

          <div className="about-vision-actions">
            <Link
              to="/translate"
              className="primary-button"
            >
              Try Sign Recognition →
            </Link>

            <Link
              to="/learn"
              className="secondary-button"
            >
              Start Learning
            </Link>
          </div>
        </div>
      </section>

      {/* Current focus */}
      <section className="about-focus">
        <div className="section-heading">
          <p className="eyebrow">CURRENT FOCUS</p>

          <h2>
            Starting with the <span>foundation.</span>
          </h2>

          <p>
            HandLex is being developed step by step, beginning
            with reliable sign recognition and a strong learning
            experience.
          </p>
        </div>

        <div className="about-focus-list">
          <div>
            <strong>01</strong>
            <span>Sign → Text recognition</span>
          </div>

          <div>
            <strong>02</strong>
            <span>Indian Sign Language learning</span>
          </div>

          <div>
            <strong>03</strong>
            <span>Real-time camera interaction</span>
          </div>

          <div>
            <strong>04</strong>
            <span>Future Text → Sign communication</span>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="about-cta">
        <p className="eyebrow">EXPLORE HANDLEX</p>

        <h2>
          Learn. Practice. <span>Communicate.</span>
        </h2>

        <p>
          Explore the platform and see where sign language and
          technology come together.
        </p>

        <div className="about-cta-actions">
          <Link
            to="/learn"
            className="primary-button"
          >
            Explore Learning →
          </Link>

          <Link
            to="/translate"
            className="secondary-button"
          >
            Try HandLex
          </Link>
        </div>
      </section>
    </main>
  );
}

export default About;