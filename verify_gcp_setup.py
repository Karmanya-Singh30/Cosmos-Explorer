#!/usr/bin/env python3
"""
Script to verify Google Cloud Platform setup for Cosmos Explorer deployment
"""

import subprocess
import sys
import os

PROJECT_ID = "cosmos-explorer-474213"

def run_command(command):
    """Run a command and return the output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_gcloud_installed():
    """Check if gcloud is installed"""
    print("Checking if gcloud is installed...")
    code, stdout, stderr = run_command("gcloud --version")
    if code == 0:
        print("✅ gcloud is installed")
        print(stdout.split('\n')[0])  # Print first line of version info
        return True
    else:
        print("❌ gcloud is not installed")
        print("Please install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        return False

def check_authentication():
    """Check if user is authenticated with gcloud"""
    print("\nChecking gcloud authentication...")
    code, stdout, stderr = run_command("gcloud auth list --format='value(account)'")
    if code == 0 and stdout.strip():
        print(f"✅ Authenticated as: {stdout.strip()}")
        return True
    else:
        print("❌ Not authenticated with gcloud")
        print("Please run: gcloud auth login")
        return False

def check_project_config():
    """Check if the correct project is configured"""
    print(f"\nChecking if project '{PROJECT_ID}' is configured...")
    code, stdout, stderr = run_command("gcloud config list --format='value(core.project)'")
    if code == 0:
        current_project = stdout.strip()
        if current_project == PROJECT_ID:
            print(f"✅ Correct project configured: {PROJECT_ID}")
            return True
        else:
            print(f"⚠️  Current project is '{current_project}', expected '{PROJECT_ID}'")
            print(f"Please run: gcloud config set project {PROJECT_ID}")
            return False
    else:
        print("❌ Unable to check project configuration")
        return False

def check_required_apis():
    """Check if required APIs are enabled"""
    print("\nChecking if required APIs are enabled...")
    required_apis = [
        "appengine.googleapis.com",
        "cloudbuild.googleapis.com",
        "containerregistry.googleapis.com"
    ]
    
    code, stdout, stderr = run_command(f"gcloud services list --project={PROJECT_ID} --format='value(config.name)'")
    if code == 0:
        enabled_apis = [line.strip() for line in stdout.split('\n') if line.strip()]
        missing_apis = [api for api in required_apis if api not in enabled_apis]
        
        if not missing_apis:
            print("✅ All required APIs are enabled")
            return True
        else:
            print("⚠️  Some required APIs are not enabled:")
            for api in missing_apis:
                print(f"  - {api}")
            print("\nEnable them with:")
            for api in missing_apis:
                print(f"  gcloud services enable {api} --project={PROJECT_ID}")
            return False
    else:
        print("❌ Unable to check API status")
        print(stderr)
        return False

def check_app_engine():
    """Check if App Engine is configured"""
    print("\nChecking App Engine configuration...")
    code, stdout, stderr = run_command(f"gcloud app describe --project={PROJECT_ID} --format='value(id)'")
    if code == 0 and stdout.strip():
        print(f"✅ App Engine is configured for project: {stdout.strip()}")
        return True
    else:
        print("⚠️  App Engine is not configured for this project")
        print(f"Configure it with: gcloud app create --project={PROJECT_ID} --region=us-central1")
        return False

def main():
    """Main function"""
    print("Cosmos Explorer - Google Cloud Platform Setup Verification")
    print("=" * 60)
    print(f"Project ID: {PROJECT_ID}")
    print()
    
    # Run all checks
    checks = [
        check_gcloud_installed,
        check_authentication,
        check_project_config,
        check_required_apis,
        check_app_engine
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 All checks passed! You're ready to deploy Cosmos Explorer to Google Cloud Platform.")
        print(f"\nTo deploy, run one of the following:")
        print(f"  1. Google App Engine: gcloud app deploy --project={PROJECT_ID}")
        print(f"  2. Or use the deployment script: deploy_to_cosmos_explorer.bat")
        return 0
    else:
        print("❌ Some checks failed. Please address the issues above before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())