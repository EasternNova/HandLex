import { useState } from "react";
import { Link } from "react-router-dom";

import "../../App.css";
import "../../styles/learn/Words.css";

function Words() {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const words = [
    {
      word: "Hello",
      category: "Greetings",
      description: "A friendly greeting used when meeting someone.",
      icon: "👋",
    },
    {
      word: "Thank You",
      category: "Common",
      description: "A way to express gratitude or appreciation.",
      icon: "🙏",
    },
    {
      word: "Please",
      category: "Common",
      description: "A polite word used when making a request.",
      icon: "🤝",
    },
    {
      word: "Sorry",
      category: "Common",
      description: "Used to express an apology.",
      icon: "💬",
    },
    {
      word: "Yes",
      category: "Responses",
      description: "Used to express agreement or confirmation.",
      icon: "👍",
    },
    {
      word: "No",
      category: "Responses",
      description: "Used to express disagreement or refusal.",
      icon: "👎",
    },
  ];

  const categories = [
    "All",
    "Greetings",
    "Common",
    "Responses",
  ];

  const filteredWords = words.filter((item) => {
    const matchesSearch = item.word
      .toLowerCase()
      .includes(searchTerm.toLowerCase());

    const matchesCategory =
      activeCategory === "All" ||
      item.category === activeCategory;

    return matchesSearch && matchesCategory;
  });

  return (
    <main className="words-page">

      {/* =========================
          HEADER
      ========================= */}

      <section className="words-header">
        <p className="eyebrow">SIGN LANGUAGE VOCABULARY</p>

        <h1>
          Learn useful <span>words.</span>
        </h1>

        <p>
          Build your everyday sign language vocabulary with
          commonly used signs and expressions.
        </p>
      </section>


      {/* =========================
          WORDS SECTION
      ========================= */}

      <section className="words-section">

        <div className="words-tools">

          <div className="words-search">
            <span>⌕</span>

            <input
              type="text"
              value={searchTerm}
              onChange={(event) =>
                setSearchTerm(event.target.value)
              }
              placeholder="Search words..."
              aria-label="Search words"
            />
          </div>


          <div className="word-filters">
            {categories.map((category) => (
              <button
                key={category}
                className={
                  activeCategory === category
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setActiveCategory(category)
                }
              >
                {category}
              </button>
            ))}
          </div>

        </div>


        {/* =========================
            WORD COUNT
        ========================= */}

        <div className="words-result-info">
          <span>
            {filteredWords.length}{" "}
            {filteredWords.length === 1
              ? "sign"
              : "signs"}
          </span>

          {searchTerm && (
            <span>
              Results for "{searchTerm}"
            </span>
          )}
        </div>


        {/* =========================
            WORD CARDS
        ========================= */}

        {filteredWords.length > 0 ? (
          <div className="words-grid">

            {filteredWords.map((item) => (
              <article
                className="word-card"
                key={item.word}
              >

                <div className="word-card-top">
                  <span>{item.category}</span>
                  <span className="word-card-number">
                    {item.word.charAt(0)}
                  </span>
                </div>


                <div className="word-sign-placeholder">
                  {item.icon}
                </div>


                <div className="word-card-content">

                  <h3>{item.word}</h3>

                  <p>{item.description}</p>

                  <Link
                    to="/practice"
                    className="secondary-button"
                  >
                    Practice →
                  </Link>

                </div>

              </article>
            ))}

          </div>
        ) : (
          <div className="words-empty">
            <div>⌕</div>

            <h3>No signs found</h3>

            <p>
              Try another word or choose a different
              category.
            </p>

            <button
              className="secondary-button"
              onClick={() => {
                setSearchTerm("");
                setActiveCategory("All");
              }}
            >
              Clear Search
            </button>
          </div>
        )}

      </section>

    </main>
  );
}

export default Words;