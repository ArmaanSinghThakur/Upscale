import React, { useState, useEffect } from "react";
import Navbar from "./Components/Navbar/Navbar";
import Home from "./Pages/Home";
import Login from "./Pages/Login";
import Upload from "./Pages/Upload";
import History from "./Pages/History";
import Contact from "./Pages/Contact";
import About from "./Pages/About";
import Profile from "./Pages/Profile"; 

const App = () => {
  const [user, setUser] = useState(null);
  const [isProfileOpen, setIsProfileOpen] = useState(false); 

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/check-auth', {
          credentials: 'include'
        });
        const data = await response.json();
        if (data.authenticated) {
          setUser(data);
        }
      } catch (error) {
        console.error("Auth check failed:", error);
      }
    };
    checkAuth();
  }, []);

  return (
    <div>
      <Navbar 
        user={user} 
        setUser={setUser} 
        onOpenProfile={() => setIsProfileOpen(true)} 
      />
      
      <Home />
      <Upload />
      <History />
      <About />
      <Contact />
      
      <Login setUser={setUser} />

      <Profile 
        isOpen={isProfileOpen} 
        onClose={() => setIsProfileOpen(false)}
        setUser={setUser}
      />

      {/* 🛠️ TEMPORARY DEBUG BUTTON 🛠️ */}
      <button
        onClick={() => setIsProfileOpen(true)}
        style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            zIndex: 9999,
            padding: '15px 25px',
            background: 'red',
            color: 'white',
            border: 'none',
            borderRadius: '50px',
            fontWeight: 'bold',
            boxShadow: '0 4px 15px rgba(255,0,0,0.4)',
            cursor: 'pointer'
        }}
      >
        🐞 Test Profile
      </button>
    </div>
  );
};

export default App;