import React, { useState, useEffect } from "react";
import "./History.css";

function History() {
  const [history, setHistory] = useState([
    { id: 1, src: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500&h=500&fit=crop&crop=face", title: "Upscaled Portrait" },
    { id: 2, src: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500&h=500&fit=crop", title: "Landscape Enhanced" },
    { id: 3, src: "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=500&h=500&fit=crop", title: "AI Restored Photo" },
    { id: 4, src: "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=500&h=500&fit=crop", title: "Old Memory Fixed" },
    { id: 1, src: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500&h=500&fit=crop&crop=face", title: "Upscaled Portrait" },
     { id: 2, src: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500&h=500&fit=crop", title: "Landscape Enhanced" },
  ]);

  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div  id ="History" className="history-container">
      {/* Background decoration */}
      <div className="bg-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>

      <div className={`content-wrapper ${isVisible ? 'visible' : ''}`}>
        <h2 className="history-title animate-slide-down">
          <span className="title-emoji">🖼️</span>
         <h1 className="main">your Image History</h1>
         </h2>
        
        <p className="history-subtitle animate-fade-in">
          All your previously enhanced images are saved here.
        </p>

        <div className="history-grid">
          {history.map((item, index) => (
            <div 
              key={item.id} 
              className="history-card animate-scale-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="card-inner">
                <div className="image-wrapper">
                  <img src={item.src} alt={item.title} className="history-img" />
                  <div className="image-overlay"></div>
                </div>
                <div className="history-overlay">
                  <div className="overlay-content">
                    <p className="overlay-title">{item.title}</p>
                    <div className="overlay-buttons">
                      <button className="btn-primary">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                          <polyline points="7,10 12,15 17,10"/>
                          <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Download
                      </button>
                      <button className="btn-secondary">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                          <circle cx="12" cy="12" r="3"/>
                        </svg>
                        View
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      
    </div>
  );
}

export default History;