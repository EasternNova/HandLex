import { Link } from "react-router-dom";

import "../../App.css";
import "../../styles/learn/Phrases.css";

function Phrases() {
  const phrases = [
    {
      phrase: "Good Morning",
      category: "Greetings",
      description:
        "A friendly phrase used when greeting someone in the morning.",
      icon: "🌅",
    },
    {
      phrase: "How Are You?",
      category: "Conversation",
      description:
        "A common phrase used to ask someone about their well-being.",
      icon: "💬",
    },
    {
      phrase: "What Is Your Name?",
      category: "Conversation",
      description:
        "Use this phrase when introducing yourself and asking someone's name.",
      icon: "👋",
    },
    {
      phrase: "Nice To Meet You",
      category: "Greetings",
      description:
        "A polite phrase used when meeting someone for the first time.",
      icon: "🤝",
    },
    {
      phrase: "See You Later",
      category: "Farewells",
      description:
        "A casual phrase used when saying goodbye for now.",
      icon: "👋",
    },
    {
      phrase: "Thank You Very Much",
      category: "Common",
      description:
        "A stronger expression of gratitude and appreciation.",
      icon: "🙏",
    },
  ];

  return (
    <main className="phrases-page">

      {/* =========================
          HEADER
      ========================= */}

      <section className="phrases-header">
        <p className="eyebrow">EVERYDAY SIGN LANGUAGE</p>

        <h1>
          Learn useful <span>phrases.</span>
        </h1>

        <p>
          Move beyond individual words and learn phrases that
          can help you communicate in everyday conversations.
        </p>
      </section>


      {/* =========================
          PHRASES
      ========================= */}

      <section className="phrases-section">

        <div className="section-heading">
          <p className="eyebrow">BUILD CONVERSATIONS</p>

          <h2>
            Start <span>communicating.</span>
          </h2>

          <p>
            Explore common phrases and practice them one step
            at a time.
          </p>
        </div>


        <div className="phrases-grid">

          {phrases.map((item) => (
            <article
              className="phrase-card"
              key={item.phrase}
            >

              <div className="phrase-card-top">
                <span>{item.category}</span>
              </div>


              <div className="phrase-sign-placeholder">
                {item.icon}
              </div>


              <div className="phrase-card-content">

                <h3>{item.phrase}</h3>

                <p>{item.description}</p>

                <Link
                  to="/practice"
                  className="secondary-button"
                >
                  Practice Phrase →
                </Link>

              </div>

            </article>
          ))}

        </div>

      </section>


      {/* =========================
          PRACTICE CTA
      ========================= */}

      <section className="phrases-cta">

        <div>
          <p className="eyebrow">READY TO PRACTICE?</p>

          <h2>
            Turn what you learn into <span>conversation.</span>
          </h2>
        </div>

        <Link
          to="/practice"
          className="primary-button"
        >
          Start Practicing →
        </Link>

      </section>

    </main>
  );
}

export default Phrases;