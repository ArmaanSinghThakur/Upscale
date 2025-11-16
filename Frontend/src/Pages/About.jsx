import React from "react";
import "./About.css";

function About() {
  return (
    <div id="About" className="about-container">
      <h1 className="about-title animate-slide-down">✨ About Upscale</h1>
      <p className="about-subtitle animate-fade-in">
        Upscale is an AI-powered image enhancement platform that restores, sharpens,
        and improves your photos with just one click.
      </p>

      <div className="about-sections">
        {/* Mission */}
        <div className="about-card animate-scale-up">
          <h2>🚀 Our Mission</h2>
          <p>
            To bring your memories to life by making every photo sharper, clearer,
            and more vibrant with the help of advanced AI.
          </p>
        </div>

        {/* Features */}
        <div className="about-card animate-scale-up">
          <h2>💡 Features</h2>
          <ul>
            <li>AI-based Image Upscaling</li>
            <li>Noise & Blur Removal</li>
            <li>Color Enhancement</li>
            <li>History of Your Uploads</li>
            <li>Secure Login & User Dashboard</li>
          </ul>
        </div>

        {/* Team */}
        <div className="about-card animate-scale-up">
          <h2>👨‍💻 Our Team</h2>
          <p>
            A group of passionate developers and designers dedicated to
            blending art with technology to give you the best photo experience.
          </p>
        </div>

        {/* Why Choose Us */}
        <div className="about-card animate-scale-up">
          <h2>🌍 Why Choose Us?</h2>
          <ul>
            <li>Fast AI-powered results in seconds</li>
            <li>Clean and user-friendly interface</li>
            <li>Cross-platform support (Web & Mobile)</li>
            <li>Free tier for casual users</li>
          </ul>
        </div>

        {/* Technologies */}
        <div className="about-card animate-scale-up">
          <h2>🛠 Technologies Used</h2>
          <p>
            We use <b>Deep Neural Networks</b>, <b>Convolutional Networks</b>,
            and <b>GANs (Generative Adversarial Networks)</b> for super-resolution.  
            The frontend is built with <b>React + Vite</b> and the backend with <b>Python (FastAPI)</b>.
          </p>
        </div>

        {/* Testimonials */}
        <div className="about-card animate-scale-up">
          <h2>⭐ What Users Say</h2>
          <blockquote>
            “Upscale turned my old blurry photos into crystal clear memories — amazing!”  
          </blockquote>
          <blockquote>
            “The best AI upscaler I’ve used. Super easy and fast!”  
          </blockquote>
        </div>

        {/* Future Vision */}
        <div className="about-card animate-scale-up">
          <h2>📈 Our Future Vision</h2>
          <p>
            We aim to bring **real-time video upscaling**, mobile app support,
            and cloud integration, making Upscale the ultimate AI photo hub.
          </p>
        </div>
      </div>
    </div>
  );
}

export default About;
