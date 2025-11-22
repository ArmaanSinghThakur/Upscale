import React, { useState } from 'react';
import './ComparisonSlider.css';

const ComparisonSlider = () => {
  const [sliderPosition, setSliderPosition] = useState(50);

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percentage = (x / rect.width) * 100;
    setSliderPosition(percentage);
  };

  const handleTouchMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
    const percentage = (x / rect.width) * 100;
    setSliderPosition(percentage);
  };

  return (
    <div className="comparison-container">
      <h3>See the Magic in Action</h3>
      <div 
        className="slider-wrapper"
        onMouseMove={handleMouseMove}
        onTouchMove={handleTouchMove}
      >
        {/* Image 2 (After) - Underneath */}
        <div className="image-after">
            {/* High-res / Upscaled Image */}
            <img 
              src="https://images.unsplash.com/photo-1494548162494-384bba4ab999?q=80&w=800&auto=format&fit=crop" 
              alt="After" 
              draggable="false" 
            />
            <span className="label">After</span>
        </div>

        {/* Image 1 (Before) - Clipped on top */}
        <div 
            className="image-before" 
            style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}
        >
            {/* Low-res / Blurry Image (Simulated with blur filter for demo if you don't have a low-res asset) */}
            <img 
              src="https://images.unsplash.com/photo-1494548162494-384bba4ab999?q=10&w=200&auto=format&fit=crop" 
              alt="Before" 
              draggable="false" 
              style={{ filter: 'blur(4px)' }}
            />
            <span className="label">Before</span>
        </div>

        {/* The Slider Handle */}
        <div 
            className="slider-handle" 
            style={{ left: `${sliderPosition}%` }}
        >
            <div className="slider-line"></div>
            <div className="slider-circle">↔</div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonSlider;