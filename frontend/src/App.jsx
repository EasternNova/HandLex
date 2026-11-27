
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Home from "./pages/Home";
import Translate from "./pages/Translate";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/translate" element={<Translate />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
