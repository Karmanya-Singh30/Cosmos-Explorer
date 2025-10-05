#!/usr/bin/env python3
"""
Script to check if all required files for Google Cloud deployment are present
"""

import os
import sys

REQUIRED_FILES = [
    'app.py',
    'app.yaml',
    'requirements.txt',
    'Dockerfile',
    'cloudbuild.yaml',
    '.gcloudignore',
    '.dockerignore'
]

REQUIRED_DIRECTORIES = [
    'templates',
    'models'
]

OPTIONAL_FILES = [
    'main.py',
    'config.py',
    'runtime.txt',
    'setup.py',
    '.env.example',
    'README.md',
    'DEPLOYMENT.md'
]

def check_files():
    """Check if all required files are present"""
    print("Checking deployment files...")
    
    missing_files = []
    present_files = []
    
    for file in REQUIRED_FILES:
        if os.path.exists(file):
            present_files.append(file)
            print(f"✓ {file} - Found")
        else:
            missing_files.append(file)
            print(f"✗ {file} - Missing")
    
    print(f"\nRequired files: {len(present_files)} present, {len(missing_files)} missing")
    
    if missing_files:
        print("\nMissing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def check_directories():
    """Check if all required directories are present"""
    print("\nChecking directories...")
    
    missing_dirs = []
    present_dirs = []
    
    for directory in REQUIRED_DIRECTORIES:
        if os.path.exists(directory) and os.path.isdir(directory):
            present_dirs.append(directory)
            print(f"✓ {directory} - Found")
        else:
            missing_dirs.append(directory)
            print(f"✗ {directory} - Missing")
    
    print(f"\nRequired directories: {len(present_dirs)} present, {len(missing_dirs)} missing")
    
    if missing_dirs:
        print("\nMissing required directories:")
        for directory in missing_dirs:
            print(f"  - {directory}")
        return False
    
    return True

def check_optional_files():
    """List optional files that are present"""
    print("\nChecking optional files...")
    
    present_optional = []
    
    for file in OPTIONAL_FILES:
        if os.path.exists(file):
            present_optional.append(file)
            print(f"✓ {file} - Found")
    
    print(f"\nOptional files present: {len(present_optional)}")

def main():
    """Main function"""
    print("Cosmos Explorer Deployment Checker")
    print("=" * 40)
    
    files_ok = check_files()
    dirs_ok = check_directories()
    check_optional_files()
    
    print("\n" + "=" * 40)
    
    if files_ok and dirs_ok:
        print("✓ All required files and directories are present!")
        print("You're ready to deploy to Google Cloud Platform.")
        return 0
    else:
        print("✗ Some required files or directories are missing.")
        print("Please check the missing items before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())