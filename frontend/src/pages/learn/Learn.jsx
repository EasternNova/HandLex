import { Link } from "react-router-dom";

import "../../App.css";
import "../../styles/learn/Learn.css";

function Learn() {
  const categories = [
    {
      icon: "🔤",
      title: "Alphabet",
      description: "Learn the hand signs for A–Z.",
      count: "26 signs",
      link: "/learn/alphabet",
    },
    {
      icon: "👋",
      title: "Common Words",
      description:
        "Build your vocabulary with everyday signs.",
      count: "6 signs",
      link: "/learn/words",
    },
    {
      icon: "💬",
      title: "Everyday Phrases",
      description:
        "Learn useful signs for real conversations.",
      count: "Coming soon",
      link: "/learn/phrases",
    },
  ];

  return (
    <main className="learn-page">

      {/* =========================
          HEADER
      ========================= */}

      <section className="learn-header">
        <p className="eyebrow">LEARN SIGN LANGUAGE</p>

        <h1>
          Learn to <span>sign.</span>
        </h1>

        <p>
          Build your sign language vocabulary step by step,
          starting with the basics and moving toward real
          communication.
        </p>
      </section>


      {/* =========================
          LEARNING PATHS
      ========================= */}

      <section className="learn-section">

        <div className="section-heading">
          <p className="eyebrow">START LEARNING</p>

          <h2>
            Choose your <span>path.</span>
          </h2>

          <p>
            Explore signs by category and learn at your own
            pace.
          </p>
        </div>


        <div className="learn-grid">

          {categories.map((category) => (
            <div
              className="learn-card"
              key={category.title}
            >

              <div className="learn-card-icon">
                {category.icon}
              </div>

              <div className="learn-card-content">

                <span>{category.count}</span>

                <h3>{category.title}</h3>

                <p>{category.description}</p>

                <Link
                  to={category.link}
                  className="primary-button"
                >
                  Start Learning →
                </Link>

              </div>

            </div>
          ))}

        </div>

      </section>


      {/* =========================
          PROGRESS
      ========================= */}

      <section className="learn-progress">

        <div>
          <p className="eyebrow">YOUR PROGRESS</p>

          <h2>
            Keep building your <span>vocabulary.</span>
          </h2>
        </div>


        <div className="progress-card">

          <div className="progress-top">
            <span>Learning progress</span>
            <strong>0%</strong>
          </div>

          <div className="progress-bar">
            <div></div>
          </div>

          <p>
            Start learning your first signs to begin tracking
            your progress.
          </p>

          <Link
            to="/learn/progress"
            className="secondary-button"
          >
            View My Progress →
          </Link>

        </div>

      </section>

    </main>
  );
}

export default Learn;