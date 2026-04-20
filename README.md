```powershell
# Create virtual environment, I isolate project dependencies
python -m venv venv

# Activate (uses isolated packages instead of global Python)
.\venv\Scripts\Activate.ps1

# Run the CNN script
python cnn.py
```

##Requirements
The virtual environment already has all required packages installed:
- TensorFlow
- NumPy
- Pillow

