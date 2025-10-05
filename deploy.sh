#!/bin/bash

# Deployment script for Cosmos Explorer to Google Cloud Platform

echo "Starting deployment of Cosmos Explorer to Google App Engine..."

# Install gcloud CLI if not already installed
# Uncomment the following lines if you need to install gcloud CLI
# echo "Installing Google Cloud SDK..."
# curl https://sdk.cloud.google.com | bash
# exec -l $SHELL
# gcloud init

# Authenticate with Google Cloud (if not already authenticated)
echo "Authenticating with Google Cloud..."
gcloud auth login

# Set the project ID (replace with your actual project ID)
echo "Setting project ID..."
read -p "Enter your Google Cloud Project ID: " PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create App Engine app if it doesn't exist
echo "Creating App Engine application if it doesn't exist..."
gcloud app create --region=us-central1

# Deploy the application
echo "Deploying the application..."
gcloud app deploy --quiet

# View the deployed application
echo "Opening the deployed application in your browser..."
gcloud app browse

echo "Deployment completed successfully!"