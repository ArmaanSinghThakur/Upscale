import React, { useState, useEffect, useRef } from "react";
import "./Contact.css";

function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
    inquiryType: "general"
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);
  const [activeField, setActiveField] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isVisible, setIsVisible] = useState(false);
  
  const contactRef = useRef(null);
  const formRef = useRef(null);

  // Intersection Observer for animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
          }
        });
      },
      { threshold: 0.1 }
    );

    if (contactRef.current) {
      observer.observe(contactRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Mouse tracking for interactive effects
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (formRef.current) {
        const rect = formRef.current.getBoundingClientRect();
        setMousePosition({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        });
      }
    };

    const formElement = formRef.current;
    if (formElement) {
      formElement.addEventListener('mousemove', handleMouseMove);
      return () => formElement.removeEventListener('mousemove', handleMouseMove);
    }
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      setSubmitStatus('success');
      setFormData({
        name: "",
        email: "",
        subject: "",
        message: "",
        inquiryType: "general"
      });
    } catch (error) {
      setSubmitStatus('error');
    } finally {
      setIsSubmitting(false);
      setTimeout(() => setSubmitStatus(null), 5000);
    }
  };

  const contactMethods = [
    {
      icon: "📧",
      title: "Email Us",
      description: "Get in touch via email",
      value: "hello@upscale.ai",
      action: "mailto:hello@upscale.ai",
      color: "from-blue-500 to-cyan-400"
    },
    {
      icon: "💬",
      title: "Live Chat",
      description: "Chat with our support team",
      value: "Available 24/7",
      action: "#",
      color: "from-green-500 to-emerald-400"
    },
    {
      icon: "📞",
      title: "Phone Support",
      description: "Call us directly",
      value: "+1 (555) 123-4567",
      action: "tel:+15551234567",
      color: "from-purple-500 to-pink-400"
    },
    {
      icon: "🌍",
      title: "Office Location",
      description: "Visit our headquarters",
      value: "San Francisco, CA",
      action: "#",
      color: "from-orange-500 to-red-400"
    }
  ];

  const faqs = [
    {
    question: "What is Upscale?",
    answer: "Upscale is a web application that uses deep neural networks to enhance low-quality images, improving resolution and clarity."
  },
  {
    question: "How do I upload an image?",
    answer: "Go to the 'Upload' section from the navbar, select your image, and click 'Upscale'."
  },
  {
    question: "Is my image safe?",
    answer: "Yes, your images are processed securely and are not stored on our servers permanently."
  },
  {
    question: "How long does it take to upscale an image?",
    answer: "Processing time depends on image size and quality, typically a few seconds to a minute."
  },
  {
    question: "Can I download the enhanced image?",
    answer: "Yes, once the upscaling is done, you can download the high-quality image directly from the result page."
  }
  ];

  const socialLinks = [
    { name: "Twitter", icon: "🐦", url: "#", color: "#1DA1F2" },
    { name: "LinkedIn", icon: "💼", url: "#", color: "#0077B5" },
    { name: "GitHub", icon: "🔧", url: "#", color: "#333" },
    { name: "Discord", icon: "💬", url: "#", color: "#5865F2" }
  ];

  return (
    <div id="Contact" ref={contactRef} className={`contact-wrapper ${isVisible ? 'loaded' : ''}`}>
      {/* Background Effects */}
      <div className="contact-bg">
        <div className="bg-gradient"></div>
        <div className="floating-particles">
          {[...Array(8)].map((_, i) => (
            <div key={i} className={`particle particle-${i + 1}`}></div>
          ))}
        </div>
      </div>

      {/* Header Section */}
      <section className="contact-header">
        <div className="header-content">
          <div className="header-badge">
            <span className="badge-pulse"></span>
            Get In Touch
          </div>
          
          <h1 className="contact-title">
            <span className="title-line">Let's Start a</span>
            <span className="title-line gradient-text">Conversation</span>
          </h1>
          
          <p className="contact-subtitle">
           "Curious about how our AI image enhancement works? We’re ready to guide you in transforming your images into stunning, high-quality visuals."
          </p>
        </div>
      </section>

      {/* Main Content */}
      <section className="contact-main">
        <div className="contact-container">
          
          {/* Contact Form */}
          <div className="form-section">
            <div className="form-header">
              <h2 className="form-title">Send us a Message</h2>
              <p className="form-subtitle">
                Fill out the form below and we'll get back to you within 24 hours.
              </p>
            </div>

            <form 
              ref={formRef}
              onSubmit={handleSubmit} 
              className="contact-form"
              style={{
                '--mouse-x': `${mousePosition.x}px`,
                '--mouse-y': `${mousePosition.y}px`
              }}
            >
              <div className="form-mouse-follower"></div>
              
              {/* Inquiry Type */}
              <div className="form-group full-width">
                <label className="form-label">What can we help you with?</label>
                <div className="inquiry-types">
                  {[
                    { value: "general", label: "General Inquiry", icon: "💡" },
                    { value: "support", label: "Technical Support", icon: "🛠️" },
                    { value: "business", label: "Business Partnership", icon: "🤝" },
                    { value: "api", label: "API Access", icon: "🔌" }
                  ].map((type) => (
                    <label key={type.value} className="inquiry-option">
                      <input
                        type="radio"
                        name="inquiryType"
                        value={type.value}
                        checked={formData.inquiryType === type.value}
                        onChange={handleInputChange}
                      />
                      <div className="option-content">
                        <span className="option-icon">{type.icon}</span>
                        <span className="option-label">{type.label}</span>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Name and Email */}
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Your Name</label>
                  <div className="input-wrapper">
                    <input
                      type="text"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      onFocus={() => setActiveField('name')}
                      onBlur={() => setActiveField(null)}
                      className={`form-input ${activeField === 'name' ? 'focused' : ''}`}
                      placeholder="Enter your full name"
                      required
                    />
                    <div className="input-highlight"></div>
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Email Address</label>
                  <div className="input-wrapper">
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      onFocus={() => setActiveField('email')}
                      onBlur={() => setActiveField(null)}
                      className={`form-input ${activeField === 'email' ? 'focused' : ''}`}
                      placeholder="your@email.com"
                      required
                    />
                    <div className="input-highlight"></div>
                  </div>
                </div>
              </div>

              {/* Subject */}
              <div className="form-group full-width">
                <label className="form-label">Subject</label>
                <div className="input-wrapper">
                  <input
                    type="text"
                    name="subject"
                    value={formData.subject}
                    onChange={handleInputChange}
                    onFocus={() => setActiveField('subject')}
                    onBlur={() => setActiveField(null)}
                    className={`form-input ${activeField === 'subject' ? 'focused' : ''}`}
                    placeholder="What's this about?"
                    required
                  />
                  <div className="input-highlight"></div>
                </div>
              </div>

              {/* Message */}
              <div className="form-group full-width">
                <label className="form-label">Message</label>
                <div className="input-wrapper">
                  <textarea
                    name="message"
                    value={formData.message}
                    onChange={handleInputChange}
                    onFocus={() => setActiveField('message')}
                    onBlur={() => setActiveField(null)}
                    className={`form-textarea ${activeField === 'message' ? 'focused' : ''}`}
                    placeholder="Tell us more about your inquiry..."
                    rows="6"
                    required
                  />
                  <div className="input-highlight"></div>
                </div>
              </div>

              {/* Submit Button */}
              <div className="form-submit">
                <button 
                  type="submit" 
                  disabled={isSubmitting}
                  className={`submit-btn ${isSubmitting ? 'submitting' : ''} ${submitStatus ? submitStatus : ''}`}
                >
                  <span className="btn-content">
                    {isSubmitting ? (
                      <>
                        <div className="spinner"></div>
                        Sending Message...
                      </>
                    ) : submitStatus === 'success' ? (
                      <>
                        <span className="success-icon">✅</span>
                        Message Sent!
                      </>
                    ) : submitStatus === 'error' ? (
                      <>
                        <span className="error-icon">❌</span>
                        Try Again
                      </>
                    ) : (
                      <>
                        <span className="send-icon">📤</span>
                        Send Message
                      </>
                    )}
                  </span>
                  <div className="btn-shine"></div>
                </button>
              </div>
            </form>
          </div>

          {/* Contact Methods */}
          <div className="methods-section">
            <div className="methods-header">
              <h2 className="methods-title">Other Ways to Reach Us</h2>
              <p className="methods-subtitle">
                Choose the method that works best for you.
              </p>
            </div>

            <div className="contact-methods">
              {contactMethods.map((method, index) => (
                <a
                  key={index}
                  href={method.action}
                  className="method-card"
                  style={{ '--delay': `${index * 0.1}s` }}
                >
                  <div className="method-icon">
                    {method.icon}
                    <div className={`method-gradient bg-gradient-to-r ${method.color}`}></div>
                  </div>
                  <div className="method-content">
                    <h3 className="method-title">{method.title}</h3>
                    <p className="method-description">{method.description}</p>
                    <div className="method-value">{method.value}</div>
                  </div>
                </a>
              ))}
            </div>

            {/* Social Links */}
            <div className="social-section">
              <h3 className="social-title">Follow Us</h3>
              <div className="social-links">
                {socialLinks.map((social, index) => (
                  <a
                    key={index}
                    href={social.url}
                    className="social-link"
                    style={{ '--social-color': social.color }}
                    title={social.name}
                  >
                    <span className="social-icon">{social.icon}</span>
                    <span className="social-name">{social.name}</span>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="faq-section">
        <div className="faq-container">
          <div className="faq-header">
            <h2 className="faq-title">Frequently Asked Questions</h2>
            <p className="faq-subtitle">
              Quick answers to common questions about our platform.
            </p>
          </div>

          <div className="faq-grid">
            {faqs.map((faq, index) => (
              <div key={index} className="faq-item" style={{ '--delay': `${index * 0.1}s` }}>
                <h3 className="faq-question">{faq.question}</h3>
                <p className="faq-answer">{faq.answer}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Office Hours */}
      <section className="hours-section">
        <div className="hours-container">
          <div className="hours-content">
            <h2 className="hours-title">We're Here When You Need Us</h2>
            <div className="hours-grid">
              <div className="hours-item">
                <div className="hours-icon">🕐</div>
                <div className="hours-info">
                  <h3>Support Hours</h3>
                  <p>Monday - Friday: 9AM - 6PM PST</p>
                  <p>Weekend: 10AM - 4PM PST</p>
                </div>
              </div>
              <div className="hours-item">
                <div className="hours-icon">⚡</div>
                <div className="hours-info">
                  <h3>Response Time</h3>
                  <p>Email: Within 24 hours</p>
                  <p>Live Chat: Within 5 minutes</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Contact;