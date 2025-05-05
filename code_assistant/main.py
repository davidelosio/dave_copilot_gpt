import subprocess
import os
import sys

def main():
    # Path to the app file
    app_file = os.path.join(os.path.dirname(__file__), "interface.py")

    # Use subprocess to invoke `streamlit run` with the app file
    try:
        subprocess.run(["streamlit", "run", app_file] + sys.argv[1:], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run Streamlit app: {e}")
