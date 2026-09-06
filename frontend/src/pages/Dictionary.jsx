import { useState } from "react";
import { Link } from "react-router-dom";

import "../App.css";
import "../styles/Dictionary.css";

function Dictionary() {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const signs = [
    {
      word: "Hello",
      category: "Greetings",
      icon: "👋",
      description: "A friendly greeting used when meeting someone.",
    },
    {
      word: "Good Morning",
      category: "Greetings",
      icon: "🌅",
      description: "A greeting commonly used in the morning.",
    },
    {
      word: "Thank You",
      category: "Common",
      icon: "🙏",
      description: "Used to express gratitude or appreciation.",
    },
    {
      word: "Please",
      category: "Common",
      icon: "🤝",
      description: "A polite expression used when making a request.",
    },
    {
      word: "Sorry",
      category: "Common",
      icon: "💬",
      description: "Used to express an apology.",
    },
    {
      word: "Yes",
      category: "Responses",
      icon: "👍",
      description: "Used to express agreement or confirmation.",
    },
    {
      word: "No",
      category: "Responses",
      icon: "👎",
      description: "Used to express disagreement or refusal.",
    },
    {
      word: "Goodbye",
      category: "Farewells",
      icon: "👋",
      description: "A phrase used when leaving or ending a conversation.",
    },
  ];

  const categories = [
    "All",
    "Greetings",
    "Common",
    "Responses",
    "Farewells",
  ];

  const filteredSigns = signs.filter((sign) => {
    const matchesSearch = sign.word
      .toLowerCase()
      .includes(searchTerm.toLowerCase());

    const matchesCategory =
      activeCategory === "All" ||
      sign.category === activeCategory;

    return matchesSearch && matchesCategory;
  });

  return (
    <main className="dictionary-page">

      {/* =========================
          HEADER
      ========================= */}

      <section className="dictionary-header">
        <p className="eyebrow">HANDLEX SIGN DICTIONARY</p>

        <h1>
          Find a <span>sign.</span>
        </h1>

        <p>
          Search HandLex vocabulary and explore signs by
          category. Your dictionary will grow as more sign
          language data is added.
        </p>
      </section>


      {/* =========================
          DICTIONARY
      ========================= */}

      <section className="dictionary-section">

        <div className="dictionary-tools">

          <div className="dictionary-search">
            <span>⌕</span>

            <input
              type="text"
              value={searchTerm}
              onChange={(event) =>
                setSearchTerm(event.target.value)
              }
              placeholder="Search signs..."
              aria-label="Search signs"
            />
          </div>


          <div className="dictionary-filters">

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
            RESULT INFO
        ========================= */}

        <div className="dictionary-result-info">
          <strong>
            {filteredSigns.length}{" "}
            {filteredSigns.length === 1
              ? "sign"
              : "signs"}
          </strong>

          {searchTerm && (
            <span>
              Results for "{searchTerm}"
            </span>
          )}
        </div>


        {/* =========================
            SIGN GRID
        ========================= */}

        {filteredSigns.length > 0 ? (
          <div className="dictionary-grid">

            {filteredSigns.map((sign) => (
              <article
                className="dictionary-card"
                key={sign.word}
              >

                <div className="dictionary-card-top">
                  <span>{sign.category}</span>

                  <span className="dictionary-letter">
                    {sign.word.charAt(0)}
                  </span>
                </div>


                <div className="dictionary-sign-placeholder">
                  {sign.icon}
                </div>


                <div className="dictionary-card-content">

                  <h3>{sign.word}</h3>

                  <p>{sign.description}</p>

                  <div className="dictionary-card-actions">

                    <Link
                      to="/practice"
                      className="secondary-button"
                    >
                      Practice →
                    </Link>

                  </div>

                </div>

              </article>
            ))}

          </div>
        ) : (
          <div className="dictionary-empty">

            <div>⌕</div>

            <h3>No signs found</h3>

            <p>
              Try another search term or select a different
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

export default Dictionary;