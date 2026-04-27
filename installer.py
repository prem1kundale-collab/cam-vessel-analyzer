import sys
import subprocess

print("Forcing installation into the correct Python environment...")

# This command tells the EXACT Python running this script to install the tools
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy', 'scikit-image', 'scipy', 'matplotlib'])

print("\n--- ALL DONE! SUCCESS! ---")
print("You can now go back to cam_analyzer.py and click Play.")