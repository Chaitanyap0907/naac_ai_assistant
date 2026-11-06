"""
setup_env.py
Run this script once to create your .env file from .env.example.
Usage:  python setup_env.py
"""
import shutil
import sys
from pathlib import Path

env_example = Path(".env.example")
env_file = Path(".env")

if env_file.exists():
    print(".env already exists. Edit it directly to update credentials.")
    sys.exit(0)

if not env_example.exists():
    print("ERROR: .env.example not found. Please re-download the project.")
    sys.exit(1)

shutil.copy(env_example, env_file)
print(".env file created from .env.example")
print()
print("Next steps:")
print("  1. Open .env in a text editor")
print("  2. Set IBM_CLOUD_API_KEY to your IBM Cloud API key")
print("  3. Set IBM_PROJECT_ID to your watsonx.ai project ID")
print("  4. Set IBM_CLOUD_URL (default: https://us-south.ml.cloud.ibm.com)")
print()
print("  Get free credentials at: https://cloud.ibm.com")
print("  watsonx.ai:              https://dataplatform.cloud.ibm.com")
