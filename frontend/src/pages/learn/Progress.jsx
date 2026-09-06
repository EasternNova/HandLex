import { Link } from "react-router-dom";

import "../../App.css";
import "../../styles/learn/Progress.css";

function Progress() {
  const stats = [
    {
      number: "0",
      label: "Signs Learned",
    },
    {
      number: "0",
      label: "Words Practiced",
    },
    {
      number: "0",
      label: "Phrases Practiced",
    },
    {
      number: "0",
      label: "Practice Sessions",
    },
  ];

  const activities = [
    {
      title: "Alphabet",
      description: "Learn the A–Z sign language alphabet.",
      progress: 0,
      link: "/learn/alphabet",
    },
    {
      title: "Common Words",
      description: "Build your everyday sign vocabulary.",
      progress: 0,
      link: "/learn/words",
    },
    {
      title: "Everyday Phrases",
      description: "Practice signs used in conversations.",
      progress: 0,
      link: "/learn/phrases",
    },
  ];

  return (
    <main className="progress-page">

      {/* =========================
          HEADER
      ========================= */}

      <section className="progress-header">
        <p className="eyebrow">YOUR LEARNING JOURNEY</p>

        <h1>
          Track your <span>progress.</span>
        </h1>

        <p>
          See what you've learned, what you've practiced,
          and where you can continue improving.
        </p>
      </section>


      {/* =========================
          STATS
      ========================= */}

      <section className="progress-stats">

        {stats.map((stat) => (
          <div className="progress-stat" key={stat.label}>
            <strong>{stat.number}</strong>
            <span>{stat.label}</span>
          </div>
        ))}

      </section>


      {/* =========================
          LEARNING PROGRESS
      ========================= */}

      <section className="progress-section">

        <div className="section-heading">
          <p className="eyebrow">LEARNING AREAS</p>

          <h2>
            Keep moving <span>forward.</span>
          </h2>

          <p>
            Continue where you left off or choose a new area
            to explore.
          </p>
        </div>


        <div className="progress-list">

          {activities.map((activity) => (
            <div
              className="progress-learning-card"
              key={activity.title}
            >

              <div className="progress-learning-info">

                <div>
                  <h3>{activity.title}</h3>

                  <p>{activity.description}</p>
                </div>

                <strong>{activity.progress}%</strong>

              </div>


              <div className="learning-progress-bar">
                <div
                  style={{
                    width: `${activity.progress}%`,
                  }}
                ></div>
              </div>


              <Link
                to={activity.link}
                className="secondary-button"
              >
                Continue Learning →
              </Link>

            </div>
          ))}

        </div>

      </section>


      {/* =========================
          START CTA
      ========================= */}

      <section className="progress-cta">

        <div>
          <p className="eyebrow">JUST GETTING STARTED?</p>

          <h2>
            Your journey begins with the <span>first sign.</span>
          </h2>
        </div>

        <Link
          to="/learn/alphabet"
          className="primary-button"
        >
          Start Learning →
        </Link>

      </section>

    </main>
  );
}

export default Progress;