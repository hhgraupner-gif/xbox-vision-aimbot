import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import Dashboard from "@/pages/Dashboard";
import Settings from "@/pages/Settings";
import ControllerSettings from "@/pages/ControllerSettings";

function App() {
  return (
    <div className="App min-h-screen bg-[#050505]">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/controller" element={<ControllerSettings />} />
        </Routes>
      </BrowserRouter>
      <Toaster 
        position="bottom-right" 
        toastOptions={{
          style: {
            background: '#0A0A0A',
            border: '1px solid rgba(255,255,255,0.1)',
            color: '#EDEDED',
          },
        }}
      />
    </div>
  );
}

export default App;
