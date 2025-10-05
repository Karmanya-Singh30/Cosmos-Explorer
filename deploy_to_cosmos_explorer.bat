@echo off
REM Deployment script for Cosmos Explorer to Google Cloud Platform
REM Specific to project: cosmos-explorer-474213

echo Starting deployment of Cosmos Explorer to Google App Engine...
echo Project: cosmos-explorer-474213

REM Authenticate with Google Cloud (if not already authenticated)
echo Authenticating with Google Cloud...
gcloud auth login

REM Set the project ID
echo Setting project ID to cosmos-explorer-474213...
gcloud config set project cosmos-explorer-474213

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