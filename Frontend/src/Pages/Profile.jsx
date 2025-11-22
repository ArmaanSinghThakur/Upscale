import React, { useState, useEffect } from "react";
import { User, Upload, Save, X, Camera, Activity } from "lucide-react";
import "./Profile.css"; // We will create this next

function Profile({ isOpen, onClose, setUser }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [newUsername, setNewUsername] = useState("");
  const [avatarFile, setAvatarFile] = useState(null);
  const [previewAvatar, setPreviewAvatar] = useState(null);
  const [message, setMessage] = useState("");

  // Fetch Profile Data when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchProfile();
    }
  }, [isOpen]);

  const fetchProfile = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/api/profile", {
        credentials: "include",
      });
      const data = await response.json();
      if (response.ok) {
        setProfile(data);
        setNewUsername(data.username);
      }
    } catch (error) {
      console.error("Error fetching profile:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAvatarChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAvatarFile(file);
      setPreviewAvatar(URL.createObjectURL(file));
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setMessage("Saving...");

    const formData = new FormData();
    formData.append("username", newUsername);
    if (avatarFile) {
      formData.append("avatar", avatarFile);
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/api/profile/update", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const data = await response.json();

      if (response.ok) {
        setMessage("Profile updated successfully!");
        // Update global user state so Navbar updates immediately
        setUser((prev) => ({ ...prev, username: data.username }));
        // Refresh local profile data to get new avatar URL
        fetchProfile();
      } else {
        setMessage(`Error: ${data.message}`);
      }
    } catch (error) {
      setMessage("Failed to update profile.");
    }
  };

  if (!isOpen) return null;

  const getAvatarUrl = (path) => {
    if (previewAvatar) return previewAvatar;
    if (path) return `http://127.0.0.1:5000${path}`;
    return "https://via.placeholder.com/150"; // Fallback
  };

  return (
    <div className="profile-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="profile-modal animate-scale-up">
        <button className="close-btn" onClick={onClose}><X size={20} /></button>
        
        <div className="profile-header">
          <h2>My Profile</h2>
          <p>Manage your account settings</p>
        </div>

        {loading ? (
          <div className="loading-spinner">Loading...</div>
        ) : (
          <div className="profile-content">
            {/* Left Column: Avatar */}
            <div className="avatar-section">
              <div className="avatar-wrapper">
                <img 
                  src={getAvatarUrl(profile?.avatar)} 
                  alt="Profile" 
                  className="profile-avatar" 
                />
                <label htmlFor="avatar-upload" className="avatar-edit-btn">
                  <Camera size={16} />
                  <input 
                    id="avatar-upload" 
                    type="file" 
                    accept="image/*" 
                    onChange={handleAvatarChange} 
                    hidden 
                  />
                </label>
              </div>
              <div className="stats-badge">
                <Activity size={14} />
                <span>{profile?.stats?.uploads || 0} Images Enhanced</span>
              </div>
            </div>

            {/* Right Column: Details Form */}
            <form className="details-form" onSubmit={handleSave}>
              <div className="form-group">
                <label>Username</label>
                <div className="input-wrapper">
                  <User size={16} className="input-icon" />
                  <input 
                    type="text" 
                    value={newUsername} 
                    onChange={(e) => setNewUsername(e.target.value)} 
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Email</label>
                <div className="input-wrapper disabled">
                  <span className="input-val">{profile?.email}</span>
                </div>
              </div>

              <div className="form-group">
                <label>Account Type</label>
                <div className="plan-badge">
                  {profile?.stats?.plan || "Free Tier"}
                </div>
              </div>

              {message && <p className={`status-msg ${message.includes("Error") ? "error" : "success"}`}>{message}</p>}

              <button type="submit" className="save-btn">
                <Save size={18} /> Save Changes
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

export default Profile;