// frontend/src/App.jsx

import React, { useState, useEffect } from "react"; // 👈 Import useState and useEffect
import Navbar from "./components/Navbar/Navbar";
import Home from "./Pages/Home";
import Login from "./Pages/Login";
import Upload from "./Pages/Upload";
import History from "./Pages/History";
import Contact from "./Pages/Contact";
import About from "./Pages/About";

const App = () => {
  // State to hold the message received from the Flask API
  const [flaskMessage, setFlaskMessage] = useState('Checking API connection...');

  // Effect to run the API fetch operation once when the component mounts
  useEffect(() => {
    // Attempt to fetch data from the CORS-enabled Flask endpoint
    // IMPORTANT: Make sure your Flask server is running on http://127.0.0.1:5000
    fetch('http://127.0.0.1:5000/api/data')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        // Success: Set the state with the message from the backend
        setFlaskMessage(`API Status: ${data.message}`);
      })
      .catch(error => {
        // Failure: Set an error message if the connection fails
        console.error('API Fetch Error:', error);
        setFlaskMessage('API Status: FAILED to connect to Flask (Check CORS/Server)');
      });
  }, []); // Empty dependency array ensures this runs only once

  return (
    <div>
      <Navbar />
      
      {/* 🟢 Display the API Status for testing purposes */}
      <p style={{ 
          textAlign: 'center', 
          padding: '10px', 
          backgroundColor: flaskMessage.includes('FAILED') ? '#fdd' : '#dfd',
          border: '1px solid #ccc'
      }}>
          {flaskMessage}
      </p>
      
      {/* Your existing page rendering structure */}
      <Home />
      <Upload/>
      <History/>
      <About/>
      <Contact/>
      <Login/>
    </div>
  );
};

export default App;