import React from "react";
import "./About.css";
import ComparisonSlider from "../Components/ComparisonSlider/ComparisonSlider"; 
import { Rocket, Lightbulb, Users, Globe, Cpu, Star, TrendingUp, Code, Server, Zap, CheckCircle } from "lucide-react";

function About() {
  return (
    <div id="About" className="about-container">
      <div className="about-header">
        <h1 className="about-title animate-slide-down">✨ About Upscale</h1>
        <p className="about-subtitle animate-fade-in">
          Upscale is an AI-powered image enhancement platform that restores, sharpens,
          and improves your photos with just one click.
        </p>
      </div>

      {/* 🖼️ "Show, Don't Tell" - Interactive Slider */}
      <div className="about-demo animate-scale-up">
        <ComparisonSlider />
      </div>

      <div className="about-sections">
        {/* Mission */}
        <div className="about-card animate-scale-up">
          <h2><Rocket className="icon" /> Our Mission</h2>
          <p>
            To bring your memories to life by making every photo sharper, clearer,
            and more vibrant with the help of advanced AI.
          </p>
        </div>

        {/* Features */}
        <div className="about-card animate-scale-up">
          <h2><Lightbulb className="icon" /> Features</h2>
          <ul className="feature-list">
            <li><CheckCircle size={16} color="#4CAF50"/> AI-based Image Upscaling</li>
            <li><CheckCircle size={16} color="#4CAF50"/> Noise & Blur Removal</li>
            <li><CheckCircle size={16} color="#4CAF50"/> Color Enhancement</li>
            <li><CheckCircle size={16} color="#4CAF50"/> History of Your Uploads</li>
            <li><CheckCircle size={16} color="#4CAF50"/> Secure Login & User Dashboard</li>
          </ul>
        </div>

        {/* Team */}
        <div className="about-card animate-scale-up">
          <h2><Users className="icon" /> Our Team</h2>
          <p>
            A group of passionate developers and designers dedicated to
            blending art with technology to give you the best photo experience.
          </p>
          <div className="team-avatars">
            <div className="avatar" title="Developer">👨‍💻</div>
            <div className="avatar" title="Designer">👩‍🎨</div>
            <div className="avatar" title="AI Engineer">🤖</div>
          </div>
        </div>

        {/* Why Choose Us */}
        <div className="about-card animate-scale-up">
          <h2><Globe className="icon" /> Why Choose Us?</h2>
          <ul className="feature-list">
            <li><Zap size={16} color="#FFD700"/> Fast AI-powered results</li>
            <li><CheckCircle size={16} color="#4CAF50"/> Clean user-friendly interface</li>
            <li><CheckCircle size={16} color="#4CAF50"/> Cross-platform support</li>
            <li><CheckCircle size={16} color="#4CAF50"/> Free tier for casual users</li>
          </ul>
        </div>

        {/* Technologies - Visualized */}
        <div className="about-card animate-scale-up">
          <h2><Cpu className="icon" /> Tech Stack</h2>
          <div className="tech-grid">
            <div className="tech-item">
                <Code className="tech-icon" color="#61DAFB" />
                <span>React + Vite</span>
            </div>
            <div className="tech-item">
                <Server className="tech-icon" color="#4CAF50" />
                <span>Python (Flask)</span>
            </div>
            <div className="tech-item">
                <Cpu className="tech-icon" color="#FF6B6B" />
                <span>PyTorch & GANs</span>
            </div>
          </div>
        </div>

        {/* Testimonials */}
        <div className="about-card animate-scale-up">
          <h2><Star className="icon" /> User Love</h2>
          <div className="testimonial">
            <p>“Upscale turned my old blurry photos into crystal clear memories — amazing!”</p>
            <span>— Alex R.</span>
          </div>
          <div className="testimonial">
            <p>“The best AI upscaler I’ve used. Super easy and fast!”</p>
            <span>— Sarah K.</span>
          </div>
        </div>

        {/* Future Vision */}
        <div className="about-card full-width animate-scale-up">
          <h2><TrendingUp className="icon" /> Our Future Vision</h2>
          <p>
            We aim to bring <strong>real-time video upscaling</strong>, mobile app support,
            and cloud integration, making Upscale the ultimate AI photo hub.
          </p>
        </div>
      </div>

      {/* 📣 Call to Action */}
      <div className="about-cta animate-fade-in">
        <h2>Ready to transform your photos?</h2>
        <a href="#Upload" className="cta-button">Start Enhancing Now</a>
      </div>
    </div>
  );
}

export default About;