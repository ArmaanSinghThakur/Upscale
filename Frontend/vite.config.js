// frontend/vite.config.js

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // New configuration section for the build output
  build: {
    // This tells Vite to output the final optimized files here:
    // fullstack-upscale/backend/dist
    outDir: '../backend/dist', 
    
    // This ensures the output directory is cleaned before a new build
    emptyOutDir: true 
  }
})