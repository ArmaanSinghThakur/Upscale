import React, { useState, useEffect } from "react";

function LoginPopup({ isOpen, onClose }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoaded, setIsLoaded] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => setIsLoaded(true), 100);
    } else {
      setIsLoaded(false);
    }
  }, [isOpen]);

  // Mouse tracking for interactive effects
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 2 - 1,
        y: (e.clientY / window.innerHeight) * 2 - 1,
      });
    };

    if (isOpen) {
      window.addEventListener('mousemove', handleMouseMove);
      return () => window.removeEventListener('mousemove', handleMouseMove);
    }
  }, [isOpen]);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when popup is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    console.log("Email:", email, "Password:", password);
    setIsSubmitting(false);
    onClose(); // Close popup after successful login
  };

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      id="log"
      className="popup-overlay"
      onClick={handleBackdropClick}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        animation: isLoaded ? 'fadeIn 0.3s ease-out' : 'none',
        padding: '20px'
      }}
    >
      <div 
        className={`login-popup ${isLoaded ? 'loaded' : ''}`}
        style={{
          position: 'relative',
          maxWidth: '400px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
          borderRadius: '24px',
          border: '1px solid rgba(148, 163, 184, 0.2)',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05) inset',
          animation: isLoaded ? 'popupSlideIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)' : 'none',
          padding: '32px'
        }}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="close-btn"
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            background: 'rgba(148, 163, 184, 0.1)',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            fontSize: '18px',
            color: '#94a3b8',
            transition: 'all 0.3s ease',
            zIndex: 10
          }}
          onMouseEnter={(e) => {
            e.target.style.background = 'rgba(239, 68, 68, 0.2)';
            e.target.style.color = '#ef4444';
            e.target.style.transform = 'scale(1.1)';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = 'rgba(148, 163, 184, 0.1)';
            e.target.style.color = '#94a3b8';
            e.target.style.transform = 'scale(1)';
          }}
        >
          ✕
        </button>

        {/* Animated Background */}
        <div 
          className="background-animation"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            borderRadius: '24px',
            overflow: 'hidden',
            pointerEvents: 'none'
          }}
        >
          <div className="floating-shapes">
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className={`shape shape-${i + 1}`}
                style={{
                  position: 'absolute',
                  background: `linear-gradient(45deg, rgba(${100 + i * 20}, ${150 + i * 15}, ${255 - i * 20}, 0.1), rgba(${200 - i * 20}, ${100 + i * 30}, ${255 - i * 10}, 0.05))`,
                  borderRadius: '50%',
                  width: `${20 + i * 10}px`,
                  height: `${20 + i * 10}px`,
                  top: `${Math.random() * 100}%`,
                  left: `${Math.random() * 100}%`,
                  animation: `float ${8 + i * 2}s infinite ease-in-out`,
                  transform: `translate(${mousePosition.x * (5 + i)}px, ${mousePosition.y * (5 + i)}px)`,
                  transition: 'transform 0.3s ease-out'
                }}
              />
            ))}
          </div>
        </div>

        {/* Login Content */}
        <div className="login-content" style={{ position: 'relative', zIndex: 2 }}>
          {/* Header */}
          <div className="login-header" style={{ textAlign: 'center', marginBottom: '32px' }}>
            <h2 style={{
              fontSize: '28px',
              fontWeight: '700',
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              marginBottom: '8px',
              animation: isLoaded ? 'slideDown 0.6s ease-out' : 'none'
            }}>
              Welcome to Upscale ✨
            </h2>
            <p style={{
              color: '#94a3b8',
              fontSize: '14px',
              animation: isLoaded ? 'slideDown 0.6s ease-out 0.2s both' : 'none'
            }}>
              Transform your images with AI magic
            </p>
          </div>

          <div onSubmit={handleSubmit}>
            {/* Email Input */}
            <div style={{
              marginBottom: '24px',
              animation: isLoaded ? 'slideLeft 0.6s ease-out 0.4s both' : 'none'
            }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: '#e2e8f0',
                fontSize: '14px',
                fontWeight: '500',
                marginBottom: '8px'
              }}>
                📧 Email Address
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: '12px',
                    color: '#e2e8f0',
                    fontSize: '14px',
                    outline: 'none',
                    transition: 'all 0.3s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#3b82f6';
                    e.target.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.1)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = 'rgba(148, 163, 184, 0.2)';
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>
            </div>

            {/* Password Input */}
            <div style={{
              marginBottom: '24px',
              animation: isLoaded ? 'slideRight 0.6s ease-out 0.6s both' : 'none'
            }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: '#e2e8f0',
                fontSize: '14px',
                fontWeight: '500',
                marginBottom: '8px'
              }}>
                🔒 Password
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  style={{
                    width: '100%',
                    padding: '12px 48px 12px 16px',
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: '12px',
                    color: '#e2e8f0',
                    fontSize: '14px',
                    outline: 'none',
                    transition: 'all 0.3s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#3b82f6';
                    e.target.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.1)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = 'rgba(148, 163, 184, 0.2)';
                    e.target.style.boxShadow = 'none';
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '16px',
                    color: '#94a3b8',
                    transition: 'color 0.3s ease'
                  }}
                >
                  {showPassword ? "👁️" : "👁️‍🗨️"}
                </button>
              </div>
            </div>

            {/* Form Options */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '32px',
              animation: isLoaded ? 'fadeIn 0.6s ease-out 0.8s both' : 'none'
            }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                color: '#94a3b8'
              }}>
                <input type="checkbox" style={{ accentColor: '#3b82f6' }} />
                Remember me
              </label>
              <a href="#forgot" style={{
                color: '#3b82f6',
                textDecoration: 'none',
                fontSize: '14px',
                transition: 'color 0.3s ease'
              }}>
                Forgot password?
              </a>
            </div>

            {/* Submit Button */}
            <button 
              type="submit" 
              disabled={isSubmitting}
              style={{
                width: '100%',
                padding: '14px',
                background: isSubmitting 
                  ? 'rgba(59, 130, 246, 0.5)' 
                  : 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                border: 'none',
                borderRadius: '12px',
                color: 'white',
                fontSize: '16px',
                fontWeight: '600',
                cursor: isSubmitting ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden',
                animation: isLoaded ? 'scaleUp 0.6s ease-out 1s both' : 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
              onMouseEnter={(e) => {
                if (!isSubmitting) {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 10px 25px rgba(59, 130, 246, 0.4)';
                }
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'none';
              }}
            >
              {isSubmitting ? '⏳ Logging in...' : '🚀 Login'}
            </button>
          </div>

          {/* Social Login */}
          <div style={{
            marginTop: '32px',
            animation: isLoaded ? 'fadeIn 0.6s ease-out 1.2s both' : 'none'
          }}>
            <div style={{
              textAlign: 'center',
              position: 'relative',
              marginBottom: '24px'
            }}>
              <div style={{
                height: '1px',
                background: 'rgba(148, 163, 184, 0.2)',
                position: 'absolute',
                top: '50%',
                left: 0,
                right: 0
              }}></div>
              <span style={{
                background: 'rgba(15, 23, 42, 0.95)',
                padding: '0 16px',
                color: '#94a3b8',
                fontSize: '14px',
                position: 'relative'
              }}>
                or continue with
              </span>
            </div>
            <div style={{ display: 'flex', gap: '12px' }}>
              <button style={{
                flex: 1,
                padding: '12px',
                background: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '12px',
                color: '#e2e8f0',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}>
                🌐 Google
              </button>
              <button style={{
                flex: 1,
                padding: '12px',
                background: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '12px',
                color: '#e2e8f0',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}>
                👨‍💻 GitHub
              </button>
            </div>
          </div>

          {/* Sign Up Link */}
          <p style={{
            textAlign: 'center',
            marginTop: '24px',
            color: '#94a3b8',
            fontSize: '14px',
            animation: isLoaded ? 'fadeIn 0.6s ease-out 1.4s both' : 'none'
          }}>
            Don't have an account? 
            <a href="#signup" style={{
              color: '#3b82f6',
              textDecoration: 'none',
              marginLeft: '4px',
              fontWeight: '500',
              transition: 'color 0.3s ease'
            }}>
              Create one →
            </a>
          </p>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes popupSlideIn {
          from {
            opacity: 0;
            transform: scale(0.9) translateY(-20px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slideLeft {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes slideRight {
          from {
            opacity: 0;
            transform: translateX(20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes scaleUp {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          33% { transform: translateY(-10px) rotate(1deg); }
          66% { transform: translateY(5px) rotate(-1deg); }
        }

        .login-popup::-webkit-scrollbar {
          width: 6px;
        }

        .login-popup::-webkit-scrollbar-track {
          background: rgba(30, 41, 59, 0.3);
          border-radius: 3px;
        }

        .login-popup::-webkit-scrollbar-thumb {
          background: rgba(59, 130, 246, 0.3);
          border-radius: 3px;
        }

        .login-popup::-webkit-scrollbar-thumb:hover {
          background: rgba(59, 130, 246, 0.5);
        }
      `}</style>
    </div>
  );
}

// Demo component to show the popup in action
function App() {
  const [isLoginOpen, setIsLoginOpen] = useState(false);

  return (
    <div id="log" style={{
      minHeight: '100vh',
    
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Demo page content */}
      <div style={{ textAlign: 'center' }}>
        <h1 style={{
          color: 'white',
          fontSize: '48px',
          marginBottom: '24px',
          background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text'
        }}>
          Upscale AI
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '18px', marginBottom: '32px' }}>
          Transform your images with cutting-edge AI technology
        </p>
        <button
          onClick={() => setIsLoginOpen(true)}
          style={{
            padding: '16px 32px',
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            border: 'none',
            borderRadius: '12px',
            color: 'white',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 15px rgba(59, 130, 246, 0.3)'
          }}
          onMouseEnter={(e) => {
            e.target.style.transform = 'translateY(-2px)';
            e.target.style.boxShadow = '0 8px 25px rgba(59, 130, 246, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = 'translateY(0)';
            e.target.style.boxShadow = '0 4px 15px rgba(59, 130, 246, 0.3)';
          }}
        >
          Open Login Page
        </button>
      </div>

      {/* Login Popup */}
      <LoginPopup 
        isOpen={isLoginOpen} 
        onClose={() => setIsLoginOpen(false)} 
      />
    </div>
  );
}

export default App;