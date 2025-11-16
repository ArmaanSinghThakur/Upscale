
import React, { useState, useEffect } from "react";
import "./Home.css";

function Home() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Sample images - you can replace with your actual imports
  const images = [
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad1' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23667eea'/%3E%3Cstop offset='100%25' stop-color='%23764ba2'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad1)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 1%3C/text%3E%3C/svg%3E",
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad2' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23f093fb'/%3E%3Cstop offset='100%25' stop-color='%23f5576c'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad2)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 2%3C/text%3E%3C/svg%3E",
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad3' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2343e97b'/%3E%3Cstop offset='100%25' stop-color='%2338f9d7'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad3)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 3%3C/text%3E%3C/svg%3E",
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad4' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23fa709a'/%3E%3Cstop offset='100%25' stop-color='%23fee140'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad4)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 4%3C/text%3E%3C/svg%3E"
  ];

  // Automatic slideshow with smooth transitions
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % images.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [images.length]);

  // Component mount animation
  useEffect(() => {
    setTimeout(() => setIsLoaded(true), 100);
  }, []);

  // Mouse tracking for parallax effect with smooth interpolation
  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = (e.clientY / window.innerHeight) * 2 - 1;
      
      setMousePosition(prev => ({
        x: prev.x + (x - prev.x) * 0.1,
        y: prev.y + (y - prev.y) * 0.1
      }));
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleSlideClick = (index) => {
    setCurrentIndex(index);
  };

  return (
    <div id="Home" className={`home-container ${isLoaded ? 'loaded' : ''}`}>
      {/* Animated Background Particles */}
      <div className="particles-bg">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className="particle"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 10}s`,
              animationDuration: `${10 + Math.random() * 20}s`
            }}
          />
        ))}
      </div>

      {/* Floating Elements */}
      <div className="floating-elements">
        <div className="floating-shape shape1" style={{
          transform: `translate(${mousePosition.x * 20}px, ${mousePosition.y * 20}px)`
        }}>✨</div>
        <div className="floating-shape shape2" style={{
          transform: `translate(${mousePosition.x * -15}px, ${mousePosition.y * -15}px)`
        }}>🎨</div>
        <div className="floating-shape shape3" style={{
          transform: `translate(${mousePosition.x * 25}px, ${mousePosition.y * 25}px)`
        }}>🚀</div>
        <div className="floating-shape shape4" style={{
          transform: `translate(${mousePosition.x * -10}px, ${mousePosition.y * -10}px)`
        }}>🔮</div>
      </div>

      {/* Main Hero Section */}
      <section className="hero">
        <div className="hero-content">
          {/* Animated Text */}
          <div className="text-container">
            <h1 className="hero-title">
              <span className="title-word word1">Transform</span>
              <span className="title-word word2">Your</span>
              <span className="title-word word3">Images</span>
              <span className="title-word word4">with</span>
              <span className="title-word word5">AI</span>
              <span className="ai-icon">🤖</span>
            </h1>
            
            <p className="hero-subtitle">
              <span className="subtitle-line line1">Upload low-quality images and witness</span>
              <span className="subtitle-line line2">the magic of AI transformation</span>
              <span className="subtitle-line line3">into stunning high-resolution masterpieces</span>
            </p>

            {/* CTA Buttons */}
            <div className="cta-buttons">
              <button className="cta-primary">
                <span>Start Enhancing</span>
                <div className="button-glow"></div>
              </button>
              <button className="cta-secondary">
                <span>Watch Demo</span>
                <div className="play-icon">▶</div>
              </button>
            </div>

            {/* Stats */}
            <div className="stats-container">
              <div className="stat-item">
                <div className="stat-number" data-target="50000">50K+</div>
                <div className="stat-label">Images Enhanced</div>
              </div>
              <div className="stat-item">
                <div className="stat-number" data-target="99">99%</div>
                <div className="stat-label">Success Rate</div>
              </div>
              <div className="stat-item">
                <div className="stat-number" data-target="24">24/7</div>
                <div className="stat-label">Processing</div>
              </div>
            </div>
          </div>

          {/* Advanced Slideshow */}
          <div className="slideshow-container">
            <div className="slideshow-wrapper">
              {/* Main Slideshow */}
              <div className="slideshow">
                {images.map((img, index) => (
                  <div
                    key={index}
                    className={`slide ${index === currentIndex ? "active" : ""} ${
                      index === (currentIndex - 1 + images.length) % images.length ? "prev" : ""
                    } ${
                      index === (currentIndex + 1) % images.length ? "next" : ""
                    }`}
                    onClick={() => handleSlideClick(index)}
                  >
                    <img src={img} alt={`Slide ${index + 1}`} />
                    <div className="slide-overlay">
                      <div className="slide-content">
                        <h3>Before & After</h3>
                        <p>AI Enhancement #{index + 1}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Slide Indicators */}
              <div className="slide-indicators">
                {images.map((_, index) => (
                  <button
                    key={index}
                    className={`indicator ${index === currentIndex ? "active" : ""}`}
                    onClick={() => handleSlideClick(index)}
                  >
                    <span className="indicator-progress"></span>
                  </button>
                ))}
              </div>

              {/* Navigation Arrows */}
              <button 
                className="nav-arrow prev-arrow"
                onClick={() => setCurrentIndex((prev) => (prev - 1 + images.length) % images.length)}
              >
                ◀
              </button>
              <button 
                className="nav-arrow next-arrow"
                onClick={() => setCurrentIndex((prev) => (prev + 1) % images.length)}
              >
                ▶
              </button>
            </div>

            {/* Progress Bar */}
            <div className="progress-container">
              <div 
                className="progress-bar"
                style={{ width: `${((currentIndex + 1) / images.length) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </section>

     
    </div>
  );
}

export default Home;











// import React, { useState, useEffect } from "react";
// import "./Home.css"

// function Home() {
//   const [currentIndex, setCurrentIndex] = useState(0);
//   const [isLoaded, setIsLoaded] = useState(false);
//   const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

//   // Sample images - you can replace with your actual imports
//   const images = [
//     "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad1' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23667eea'/%3E%3Cstop offset='100%25' stop-color='%23764ba2'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad1)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 1%3C/text%3E%3C/svg%3E",
//     "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad2' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23f093fb'/%3E%3Cstop offset='100%25' stop-color='%23f5576c'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad2)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 2%3C/text%3E%3C/svg%3E",
//     "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad3' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2343e97b'/%3E%3Cstop offset='100%25' stop-color='%2338f9d7'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad3)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 3%3C/text%3E%3C/svg%3E",
//     "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='grad4' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23fa709a'/%3E%3Cstop offset='100%25' stop-color='%23fee140'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23grad4)'/%3E%3Ctext x='200' y='150' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3EImage 4%3C/text%3E%3C/svg%3E"
//   ];

//   // Automatic slideshow with smooth transitions
//   useEffect(() => {
//     const interval = setInterval(() => {
//       setCurrentIndex((prev) => (prev + 1) % images.length);
//     }, 4000);
//     return () => clearInterval(interval);
//   }, [images.length]);

//   // Component mount animation
//   useEffect(() => {
//     setTimeout(() => setIsLoaded(true), 100);
//   }, []);

//   /*Mouse tracking for parallax effect*/
//   useEffect(() => {
//     const handleMouseMove = (e) => {
//       setMousePosition({
//         x: (e.clientX / window.innerWidth) * 2 - 1,
//         y: (e.clientY / window.innerHeight) * 2 - 1,
//       });
//     };

//     window.addEventListener('mousemove', handleMouseMove);
//     return () => window.removeEventListener('mousemove', handleMouseMove);
//   }, []);

//   const handleSlideClick = (index) => {
//     setCurrentIndex(index);
//   };

//   return (
//     <div className={`home-container ${isLoaded ? 'loaded' : ''}`}>
//       {/* Animated Background Particles */}
//       <div className="particles-bg">
//         {[...Array(50)].map((_, i) => (
//           <div
//             key={i}
//             className="particle"
//             style={{
//               left: `${Math.random() * 100}%`,
//               animationDelay: `${Math.random() * 10}s`,
//               animationDuration: `${10 + Math.random() * 20}s`
//             }}
//           />
//         ))}
//       </div>

//       {/* Floating Elements */}
//       <div className="floating-elements">
//         <div className="floating-shape shape1" style={{
//           transform: `translate(${mousePosition.x * 20}px, ${mousePosition.y * 20}px)`
//         }}>✨</div>
//         <div className="floating-shape shape2" style={{
//           transform: `translate(${mousePosition.x * -15}px, ${mousePosition.y * -15}px)`
//         }}>🎨</div>
//         <div className="floating-shape shape3" style={{
//           transform: `translate(${mousePosition.x * 25}px, ${mousePosition.y * 25}px)`
//         }}>🚀</div>
//         <div className="floating-shape shape4" style={{
//           transform: `translate(${mousePosition.x * -10}px, ${mousePosition.y * -10}px)`
//         }}>🔮</div>
//       </div>

//       {/* Main Hero Section */}
//       <section className="hero">
//         <div className="hero-content">
//           {/* Animated Text */}
//           <div className="text-container">
//             <h1 className="hero-title">
//               <span className="title-word word1">Transform</span>
//               <span className="title-word word2">Your</span>
//               <span className="title-word word3">Images</span>
//               <span className="title-word word4">with</span>
//               <span className="title-word word5">AI</span>
//               <span className="ai-icon">🤖</span>
//             </h1>
            
//             <p className="hero-subtitle">
//               <span className="subtitle-line line1">Upload low-quality images and witness</span>
//               <span className="subtitle-line line2">the magic of AI transformation</span>
//               <span className="subtitle-line line3">into stunning high-resolution masterpieces</span>
//             </p>

//             {/* CTA Buttons */}
//             <div className="cta-buttons">
//               <button className="cta-primary">
//                 <span>Start Enhancing</span>
//                 <div className="button-glow"></div>
//               </button>
//               <button className="cta-secondary">
//                 <span>Watch Demo</span>
//                 <div className="play-icon">▶</div>
//               </button>
//             </div>

//             {/* Stats */}
//             <div className="stats-container">
//               <div className="stat-item">
//                 <div className="stat-number" data-target="50000">0</div>
//                 <div className="stat-label">Images Enhanced</div>
//               </div>
//               <div className="stat-item">
//                 <div className="stat-number" data-target="99">0</div>
//                 <div className="stat-label">% Success Rate</div>
//               </div>
//               <div className="stat-item">
//                 <div className="stat-number" data-target="24">0</div>
//                 <div className="stat-label">Hour Processing</div>
//               </div>
//             </div>
//           </div>

//           {/* Advanced Slideshow */}
//           <div className="slideshow-container">
//             <div className="slideshow-wrapper">
//               {/* Main Slideshow */}
//               <div className="slideshow">
//                 {images.map((img, index) => (
//                   <div
//                     key={index}
//                     className={`slide ${index === currentIndex ? "active" : ""} ${
//                       index === (currentIndex - 1 + images.length) % images.length ? "prev" : ""
//                     } ${
//                       index === (currentIndex + 1) % images.length ? "next" : ""
//                     }`}
//                     onClick={() => handleSlideClick(index)}
//                   >
//                     <img src={img} alt={`Slide ${index + 1}`} />
//                     <div className="slide-overlay">
//                       <div className="slide-content">
//                         <h3>Before & After</h3>
//                         <p>AI Enhancement #{index + 1}</p>
//                       </div>
//                     </div>
//                   </div>
//                 ))}
//               </div>

//               {/* Slide Indicators */}
//               { <div className="slide-indicators">
//                 {images.map((_, index) => (
//                   <button
//                     key={index}
//                     className={`indicator ${index === currentIndex ? "active" : ""}`}
//                     onClick={() => handleSlideClick(index)}
//                   >
//                     <span className="indicator-progress"></span>
//                   </button>
//                 ))}
//               </div> }

//               {/* Navigation Arrows */}
//               <button 
//                 className="nav-arrow prev-arrow"
//                 onClick={() => setCurrentIndex((prev) => (prev - 1 + images.length) % images.length)}
//               >
//                 ◀
//               </button>
//               <button 
//                 className="nav-arrow next-arrow"
//                 onClick={() => setCurrentIndex((prev) => (prev + 1) % images.length)}
//               >
//                 ▶
//               </button>
//             </div>

//             {/* Progress Bar */}
//             <div className="progress-container">
//               <div 
//                 className="progress-bar"
//                 style={{ width: `${((currentIndex + 1) / images.length) * 100}%` }}
//               ></div>
//             </div>
//           </div>
//         </div>


//       </section>

//       <style jsx>{`
//         * {
//           margin: 0;
//           padding: 0;
//           box-sizing: border-box;
//         }

//         .home-container {
//           min-height: 100vh;
//           position: relative;
//           background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
//           overflow-x: hidden;
//           font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
//           opacity: 0;
//           transition: opacity 1s ease-out;
//         }

//         .home-container.loaded {
//           opacity: 1;
//         }

//         /* Particles Background */
//         .particles-bg {
//           position: absolute;
//           top: 0;
//           left: 0;
//           width: 100%;
//           height: 100%;
//           overflow: hidden;
//           z-index: 1;
//         }

//         .particle {
//           position: absolute;
//           width: 4px;
//           height: 4px;
//           background: rgba(255, 255, 255, 0.6);
//           border-radius: 50%;
//           animation: floatUp infinite linear;
//         }

//         @keyframes floatUp {
//           from {
//             transform: translateY(100vh) translateX(0);
//             opacity: 0;
//           }
//           10% {
//             opacity: 1;
//           }
//           90% {
//             opacity: 1;
//           }
//           to {
//             transform: translateY(-100px) translateX(50px);
//             opacity: 0;
//           }
//         }

//         /* Floating Elements */
//         .floating-elements {
//           position: absolute;
//           top: 0;
//           left: 0;
//           width: 100%;
//           height: 100%;
//           pointer-events: none;
//           z-index: 2;
//         }

//         .floating-shape {
//           position: absolute;
//           font-size: 2rem;
//           animation: float 6s ease-in-out infinite;
//           transition: transform 0.3s ease;
//         }

//         .shape1 { top: 20%; left: 10%; animation-delay: 0s; }
//         .shape2 { top: 60%; right: 10%; animation-delay: 2s; }
//         .shape3 { bottom: 30%; left: 15%; animation-delay: 4s; }
//         .shape4 { top: 40%; right: 20%; animation-delay: 1s; }

//         @keyframes float {
//           0%, 100% { transform: translateY(0px) rotate(0deg); }
//           50% { transform: translateY(-20px) rotate(180deg); }
//         }

//         /* Hero Section */
//         .hero {
//           position: relative;
//           min-height: 50vh;
//           display: flex;
//           align-items: center;
//           justify-content: center;
//           z-index: 3;
//           padding: 2rem;
//         }

//         .hero-content {
//           display: grid;
//           grid-template-columns: 1fr 1fr;
//           gap: 4rem;
//           max-width: 1200px;
//           width: 100%;
//           align-items: center;
//         }

//         /* Text Container */
//         .text-container {
//           text-align: left;
//         }

//         .hero-title {
//           font-size: 4rem;
//           font-weight: 800;
//           line-height: 1.1;
//           margin-bottom: 1.5rem;
//           display: flex;
//           flex-wrap: wrap;
//           gap: 0.5rem;
//           align-items: center;
//         }

//         .title-word {
//           display: inline-block;
//           background: linear-gradient(135deg, #fff, #f0f8ff);
//           -webkit-background-clip: text;
//           -webkit-text-fill-color: transparent;
//           background-clip: text;
//           animation: wordSlide 0.8s ease-out forwards;
//           opacity: 0;
//           transform: translateY(30px);
//         }

//         .word1 { animation-delay: 0.1s; }
//         .word2 { animation-delay: 0.2s; }
//         .word3 { animation-delay: 0.3s; }
//         .word4 { animation-delay: 0.4s; }
//         .word5 { animation-delay: 0.5s; }

//         .ai-icon {
//           font-size: 3rem;
//           animation: iconBounce 2s infinite, wordSlide 0.8s 0.6s ease-out forwards;
//           opacity: 0;
//           transform: translateY(30px);
//         }

//         @keyframes wordSlide {
//           to {
//             opacity: 1;
//             transform: translateY(0);
//           }
//         }

//         @keyframes iconBounce {
//           0%, 20%, 50%, 80%, 100% {
//             transform: translateY(0);
//           }
//           40% {
//             transform: translateY(-10px);
//           }
//           60% {
//             transform: translateY(-5px);
//           }
//         }

//         .hero-subtitle {
//           font-size: 1.25rem;
//           color: rgba(255, 255, 255, 0.9);
//           line-height: 1.6;
//           margin-bottom: 2.5rem;
//           display: flex;
//           flex-direction: column;
//           gap: 0.5rem;
//         }

//         .subtitle-line {
//           opacity: 0;
//           transform: translateX(-30px);
//           animation: lineSlide 0.8s ease-out forwards;
//         }

//         .line1 { animation-delay: 0.8s; }
//         .line2 { animation-delay: 1s; }
//         .line3 { animation-delay: 1.2s; }

//         @keyframes lineSlide {
//           to {
//             opacity: 1;
//             transform: translateX(0);
//           }
//         }

//         /* CTA Buttons */
//         .cta-buttons {
//           display: flex;
//           gap: 1rem;
//           margin-bottom: 3rem;
//           opacity: 0;
//           animation: buttonsFade 0.8s 1.4s ease-out forwards;
//         }

//         @keyframes buttonsFade {
//           to { opacity: 1; }
//         }

//         .cta-primary, .cta-secondary {
//           padding: 1rem 2rem;
//           border: none;
//           border-radius: 50px;
//           font-size: 1.1rem;
//           font-weight: 600;
//           cursor: pointer;
//           transition: all 0.3s ease;
//           position: relative;
//           overflow: hidden;
//           display: flex;
//           align-items: center;
//           gap: 0.5rem;
//         }

//         .cta-primary {
//           background: linear-gradient(135deg, #ff6b6b, #ff8e53);
//           color: white;
//           box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
//         }

//         .cta-primary:hover {
//           transform: translateY(-2px);
//           box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
//         }

//         .cta-secondary {
//           background: rgba(255, 255, 255, 0.1);
//           color: white;
//           border: 2px solid rgba(255, 255, 255, 0.3);
//           backdrop-filter: blur(10px);
//         }

//         .cta-secondary:hover {
//           background: rgba(255, 255, 255, 0.2);
//           transform: translateY(-2px);
//         }

//         .button-glow {
//           position: absolute;
//           top: 0;
//           left: -100%;
//           width: 100%;
//           height: 100%;
//           background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
//           transition: left 0.5s;
//         }

//         .cta-primary:hover .button-glow {
//           left: 100%;
//         }

//         .play-icon {
//           width: 20px;
//           height: 20px;
//           background: rgba(255, 255, 255, 0.2);
//           border-radius: 50%;
//           display: flex;
//           align-items: center;
//           justify-content: center;
//           font-size: 0.8rem;
//         }

//         /* Stats */
//         .stats-container {
//           display: flex;
//           gap: 2rem;
//           opacity: 0;
//           animation: statsFade 0.8s 1.6s ease-out forwards;
//         }

//         @keyframes statsFade {
//           to { opacity: 1; }
//         }

//         .stat-item {
//           text-align: center;
//         }

//         .stat-number {
//           font-size: 2.5rem;
//           font-weight: 800;
//           color: white;
//           line-height: 1;
//         }

//         .stat-label {
//           font-size: 0.9rem;
//           color: rgba(255, 255, 255, 0.8);
//           margin-top: 0.5rem;
//         }

//         /* Slideshow Container */
//         .slideshow-container {
//           position: relative;
//           opacity: 0;
//           transform: translateX(10px);
//           animation: slideshowEnter 0s 0.1s ease-out forwards;
//         }

//         @keyframes slideshowEnter {
//           to {
//             opacity: 1;
//             transform: translateX(0);
//           }
//         }

//         .slideshow-wrapper {
//           position: relative;
//           width: 500px;
//           height: 350px;
//           border-radius: 20px;
//           overflow: hidden;
//           box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
//         }

//         .slideshow {
//           position: relative;
//           width: 100%;
//           height: 100%;
//         }

//         .slide {
//           position: absolute;
//           top: 0;
//           left: 0;
//           width: 100%;
//           height: 100%;
//           opacity: 0;
//           transform: scale(1.1);
//           transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
//           cursor: pointer;
//         }

//         .slide.active {
//           opacity: 1;
//           transform: scale(1);
//           z-index: 2;
//         }

//         .slide.prev {
//           opacity: 0.3;
//           transform: scale(0.95) translateX(-20%);
//           z-index: 1;
//         }

//         .slide.next {
//           opacity: 0.3;
//           transform: scale(0.95) translateX(20%);
//           z-index: 1;
//         }

//         .slide img {
//           width: 100%;
//           height: 100%;
//           object-fit: cover;
//           transition: transform 0.3s ease;
//         }

//         .slide:hover img {
//           transform: scale(1.05);
//         }

//         .slide-overlay {
//           position: absolute;
//           bottom: 0;
//           left: 0;
//           right: 0;
//           background: linear-gradient(transparent, rgba(0,0,0,0.8));
//           padding: 2rem;
//           transform: translateY(100%);
//           transition: transform 0.3s ease;
//         }

//         .slide.active .slide-overlay {
//           transform: translateY(0);
//         }

//         .slide-content h3 {
//           color: white;
//           font-size: 1.5rem;
//           margin-bottom: 0.5rem;
//         }

//         .slide-content p {
//           color: rgba(255, 255, 255, 0.8);
//         }

//         /* Navigation */
//         .nav-arrow {
//           position: absolute;
//           top: 50%;
//           transform: translateY(-50%);
//           background: rgba(255, 255, 255, 0.1);
//           border: none;
//           width: 50px;
//           height: 50px;
//           border-radius: 50%;
//           color: white;
//           font-size: 1.2rem;
//           cursor: pointer;
//           transition: all 0.3s ease;
//           backdrop-filter: blur(10px);
//           z-index: 3;
//         }

//         .prev-arrow {
//           left: -25px;
//         }

//         .next-arrow {
//           right: -25px;
//         }

//         .nav-arrow:hover {
//           background: rgba(255, 255, 255, 0.2);
//           transform: translateY(-50%) scale(1.1);
//         }

//         /* Indicators */
//         .slide-indicators {
//           position: absolute;
//           bottom: 20px;
//           left: 50%;
//           transform: translateX(-50%);
//           display: flex;
//           gap: 10px;
//           z-index: 3;
//         }

//         .indicator {
//           width: 40px;
//           height: 4px;
//           background: rgba(255, 255, 255, 0.3);
//           border: none;
//           border-radius: 2px;
//           cursor: pointer;
//           overflow: hidden;
//           position: relative;
//         }

//         .indicator-progress {
//           position: absolute;
//           top: 0;
//           left: 0;
//           height: 100%;
//           background: white;
//           width: 0;
//           transition: width 0.3s ease;
//         }

//         .indicator.active .indicator-progress {
//           width: 100%;
//           animation: progressFill 4s linear;
//         }

//         @keyframes progressFill {
//           from { width: 0; }
//           to { width: 100%; }
//         }

//         /* Progress Bar */
//         .progress-container {
//           width: 100%;
//           height: 2px;
//           background: rgba(255, 255, 255, 0.2);
//           margin-top: 20px;
//           border-radius: 1px;
//           overflow: hidden;
//         }

//         .progress-bar {
//           height: 100%;
//           background: linear-gradient(90deg, #ff6b6b, #ff8e53);
//           border-radius: 1px;
//           transition: width 0.8s ease;
//         }



//         /* Responsive Design */
//         @media (max-width: 1024px) {
//           .hero-content {
//             grid-template-columns: 1fr;
//             gap: 3rem;
//             text-align: center;
//           }

//           .slideshow-wrapper {
//             width: 100%;
//             max-width: 450px;
//             height: 300px;
//           }

//           .hero-title {
//             font-size: 3rem;
//             justify-content: center;
//           }
//         }

//         @media (max-width: 768px) {
//           .hero {
//             padding: 1rem;
//           }

//           .hero-title {
//             font-size: 2.5rem;
//           }

//           .hero-subtitle {
//             font-size: 1.1rem;
//           }

//           .cta-buttons {
//             flex-direction: column;
//             align-items: center;
//           }

//           .cta-primary, .cta-secondary {
//             width: 100%;
//             max-width: 280px;
//             justify-content: center;
//           }

//           .stats-container {
//             flex-direction: column;
//             gap: 1rem;
//           }

//           .slideshow-wrapper {
//             height: 250px;
//           }

//           .nav-arrow {
//             width: 40px;
//             height: 40px;
//             font-size: 1rem;
//           }

//           .prev-arrow {
//             left: -20px;
//           }

//           .next-arrow {
//             right: -20px;
//           }
//         }
//       `}</style>
//     </div>
//   );
// }

// export default Home;