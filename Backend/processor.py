import os
import torch
# Import the helper functions from our new upscale module
try:
    from upscale import load_model, upscale_image
except ImportError:
    from .upscale import load_model, upscale_image

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'final_model.pth') 
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'uploads', 'processed')

# Ensure output folder exists
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# --- INITIALIZE MODEL ONCE ---
# Load the model globally so it stays in memory
# We pass just the path, the device logic is inside upscale.py
loaded_model = load_model(MODEL_PATH)

def process_image_for_upscale(unique_filename_server):
    """
    Wrapper function called by app.py.
    Delegates the actual work to upscale.py's functions.
    """
    if loaded_model is None:
        print("Cannot process: Model failed to load.")
        return False

    originals_path = os.path.join(BASE_DIR, 'uploads', 'originals', unique_filename_server)
    processed_path = os.path.join(PROCESSED_FOLDER, unique_filename_server)
    
    # Call the imported function
    success = upscale_image(loaded_model, originals_path, processed_path)
    
    return success