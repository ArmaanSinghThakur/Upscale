import React, { useState } from "react";
import "./Upload.css";

function Upload() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(URL.createObjectURL(e.target.files[0]));
    }
  };

  const handleUpload = async () => {
    if (selectedImage) {
      setIsUploading(true);
      console.log("Uploading image...");
      
      // Simulate upload delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      alert("Image uploaded successfully!");
      setIsUploading(false);
    } else {
      alert("Please select an image first.");
    }
  };

  // Sample image URLs for floating background
  const floatingImages = [
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='80' viewBox='0 0 120 80'%3E%3Crect width='120' height='80' fill='%23e0f2fe'/%3E%3Crect x='10' y='20' width='100' height='40' fill='%23b3e5fc' rx='5'/%3E%3Ccircle cx='25' cy='30' r='8' fill='%23ffeb3b'/%3E%3Cpath d='M40 50L60 35L80 45L100 40L100 60L40 60Z' fill='%234caf50'/%3E%3C/svg%3E",
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='80' viewBox='0 0 120 80'%3E%3Crect width='120' height='80' fill='%23fce4ec'/%3E%3Crect x='10' y='20' width='100' height='40' fill='%23f8bbd9' rx='5'/%3E%3Ccircle cx='30' cy='35' r='6' fill='%23ff9800'/%3E%3Cpath d='M45 50L65 30L85 40L95 35L95 60L45 60Z' fill='%239c27b0'/%3E%3C/svg%3E",
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='80' viewBox='0 0 120 80'%3E%3Crect width='120' height='80' fill='%23e8f5e8'/%3E%3Crect x='15' y='15' width='90' height='50' fill='%23a5d6a7' rx='5'/%3E%3Ccircle cx='35' cy='30' r='7' fill='%23ffeb3b'/%3E%3Cpath d='M50 50L70 35L85 45L95 40L95 60L50 60Z' fill='%232196f3'/%3E%3C/svg%3E",
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='80' viewBox='0 0 120 80'%3E%3Crect width='120' height='80' fill='%23fff3e0'/%3E%3Crect x='10' y='15' width='100' height='50' fill='%23ffcc02' rx='5'/%3E%3Ccircle cx='25' cy='28' r='8' fill='%23f44336'/%3E%3Cpath d='M40 55L60 40L80 50L100 45L100 65L40 65Z' fill='%23ff5722'/%3E%3C/svg%3E"
  ];

  return (
    <div  id ="Upload" className="upload-container">
      {/* Animated Background with Floating Images */}
      <div className="floating-bg">
        {Array.from({ length: 20 }, (_, i) => (
          <div
            key={i}
            className="floating-image"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 10}s`,
              animationDuration: `${15 + Math.random() * 10}s`
            }}
          >
            <img 
              src={floatingImages[i % floatingImages.length]} 
              alt=""
              style={{
                width: `${60 + Math.random() * 40}px`,
                opacity: 0.1 + Math.random() * 0.2
              }}
            />
          </div>
        ))}
      </div>

      {/* Gradient Overlay */}
      <div className="gradient-overlay"></div>

      {/* Main Upload Box */}
      <div className="upload-box">
        <div className="upload-header">
          <h1 className="main-title">✨ AI Image Enhancer</h1>
          <p className="subtitle">Transform your images with cutting-edge AI technology</p>
        </div>

        {/* Upload Area */}
        <div className="upload-section">
          <label htmlFor="file-upload" className="upload-label">
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              style={{ display: 'none' }}
            />
            <div className="upload-icon">📸</div>
            <span className="upload-text">Choose Your Image</span>
            <div className="upload-hint">PNG, JPG, WEBP up to 10MB</div>
          </label>

          {/* Image Preview */}
          {selectedImage && (
            <div className="preview-container">
              <div className="preview-box">
                <img src={selectedImage} alt="Preview" className="preview-img" />
                <div className="preview-overlay">
                  <span>Ready for Enhancement</span>
                </div>
              </div>
            </div>
          )}

          {/* Upload Button */}
          <button 
            className={`upload-btn ${isUploading ? 'uploading' : ''}`}
            onClick={handleUpload}
            disabled={isUploading}
          >
            {isUploading ? (
              <>
                <div className="spinner"></div>
                Processing...
              </>
            ) : (
              <>
                🚀 Enhance Image
              </>
            )}
          </button>
        </div>

        {/* Features */}
        <div className="features">
          <div className="feature">
            <span className="feature-icon">⚡</span>
            <span>Lightning Fast</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🎯</span>
            <span>AI Powered</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🔒</span>
            <span>Secure Upload</span>
          </div>
        </div>
      </div>

      
    </div>
  );
}

export default Upload;