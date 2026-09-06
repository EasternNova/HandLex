import "../../App.css";
import "../../styles/learn/Alphabet.css";

function Alphabet() {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

  return (
    <main className="alphabet-page">

      <section className="alphabet-header">
        <p className="eyebrow">SIGN LANGUAGE ALPHABET</p>

        <h1>
          Learn the <span>alphabet.</span>
        </h1>

        <p>
          Explore the hand signs for A–Z and build the foundation
          for your sign language vocabulary.
        </p>
      </section>

      <section className="alphabet-section">

        <div className="alphabet-grid">
          {letters.map((letter) => (
            <div className="alphabet-card" key={letter}>
              <div className="letter">
                {letter}
              </div>

              <div className="sign-placeholder">
                🤟
              </div>

              <h3>{letter}</h3>

              <button className="secondary-button">
                Practice →
              </button>
            </div>
          ))}
        </div>

      </section>

    </main>
  );
}

export default Alphabet;