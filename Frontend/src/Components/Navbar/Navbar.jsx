import React, { useState } from 'react';
import { Menu, X, Home, History, Info, Upload, Mail, LogIn } from 'lucide-react';
import './Navbar.css';
import AnchorLink from "react-anchor-link-smooth-scroll";




const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const navItems = [
    { name: 'Home', href: '#Home', icon: Home },
    { name: 'History', href: '#History', icon: History },
    { name: 'About', href: '#About', icon: Info },
    { name: 'Upload', href: '#Upload', icon: Upload },
    { name: 'Contact', href: '#Contact', icon: Mail }
    
  ];








  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="navbar-container">
        <div className="navbar-logo">
          <div className="brand-box">US</div>
          <h1 className="brand-title">Upscale</h1>
        </div>

        {/* Desktop Links */}
        <div className="navbar-links">
          {navItems.map(item => {
            const Icon = item.icon;
            return (
              <a key={item.name} href={item.href} className="nav-link">
                <Icon size={18} />
                <span>{item.name}</span>
              </a>
            );
          })}

          {/* Login Button */}
          <a href="#log" className="nav-link login-link">
            <LogIn size={18} />
          
            
             <AnchorLink
            className="anchor-link"
            href="#log"
            offset={50}
            onClick={() => {
              setMenu("Login");
              handleClose();
            }}
          >
            <span>Log in</span>
          </AnchorLink>
          
             
              



          </a>
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="menu-toggle"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-label="Toggle menu"
        >
          {isOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="mobile-menu">
          {navItems.map(item => {
            const Icon = item.icon;
            return (
              <a
                key={item.name}
                href={item.href}
                className="mobile-link"
                onClick={() => setIsOpen(false)}
              >
                <Icon size={20} />
                <span>{item.name}</span>
              </a>
            );
          })}
          {/* Mobile Login */}
          <a href="#log" className="mobile-link login-mobile" onClick={() => setIsOpen(false)}>
            <LogIn size={20} />
             <AnchorLink
            className="anchor-link"
            href="#aaa"
            offset={50}
            onClick={() => {
              setMenu("Login");
              handleClose();
            }}
          >
            <span>Log in</span>
          </AnchorLink>
          </a>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
