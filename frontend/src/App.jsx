import { BrowserRouter, Routes, Route } from "react-router-dom";

import Home from "./pages/Home";
import Translate from "./pages/Translate";

import Learn from "./pages/learn/Learn";
import Alphabet from "./pages/learn/Alphabet";
import Words from "./pages/learn/Words";
import Phrases from "./pages/learn/Phrases";
import Progress from "./pages/learn/Progress";

import Practice from "./pages/Practice";
import Dictionary from "./pages/Dictionary";
import About from "./pages/About";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />

        <Route path="/translate" element={<Translate />} />

        <Route path="/learn" element={<Learn />} />
        <Route path="/learn/alphabet" element={<Alphabet />} />
        <Route path="/learn/words" element={<Words />} />
        <Route path="/learn/phrases" element={<Phrases />} />
        <Route path="/learn/progress" element={<Progress />} />

        <Route path="/practice" element={<Practice />} />
        <Route path="/dictionary" element={<Dictionary />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;