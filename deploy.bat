@echo off
REM Deployment script for Cosmos Explorer to Google Cloud Platform

echo Starting deployment of Cosmos Explorer to Google App Engine...

REM Authenticate with Google Cloud (if not already authenticated)
echo Authenticating with Google Cloud...
gcloud auth login

REM Set the project ID (replace with your actual project ID)
echo Setting project ID...
set /p PROJECT_ID="Enter your Google Cloud Project ID: "
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo Enabling required APIs...
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com

REM Create App Engine app if it doesn't exist
echo Creating App Engine application if it doesn't exist...
gcloud app create --region=us-central1

REM Deploy the application
echo Deploying the application...
gcloud app deploy --quiet

REM View the deployed application
echo Opening the deployed application in your browser...
gcloud app browse

echo Deployment completed successfully!
pause