# Cosmos Explorer

Cosmos Explorer is an AI-powered exoplanet detection and classification platform that uses machine learning to identify potential exoplanets from astronomical datasets collected by NASA's Kepler, K2, and TESS missions.

## Overview

The project automates the analysis of large-scale astronomical datasets by processing planetary features such as orbital period, transit duration, and planetary radius. Machine learning models are used to classify celestial objects as confirmed exoplanets, candidates, or false positives.

## Key Features

1. Upload and analyze exoplanet datasets from NASA missions
2. Machine learning-based exoplanet classification
3. Interactive dashboard for visualization and exploration
4. Real-time prediction and model evaluation
5. Automated processing of large astronomical datasets

## Tech Stack

Python
Flask
Pandas
NumPy
Scikit-Learn / Deep Learning Models
HTML, CSS, JavaScript
Google Cloud Platform

## Applications

Astronomical research
Space science education
Exoplanet candidate screening
Data-driven astrophysics studies

## Local Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cosmos-explorer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at http://localhost:8080

## Google Cloud Deployment

### Prerequisites for Google Cloud Deployment

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Create a Google Cloud Project
3. Enable billing for your project

### Deployment Steps

1. Install and initialize the Google Cloud SDK:
   ```
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. Authenticate with Google Cloud:
   ```
   gcloud auth login
   ```

3. Set your project ID:
   ```
   gcloud config set project YOUR_PROJECT_ID
   ```

4. Enable required APIs:
   ```
   gcloud services enable appengine.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

5. Create an App Engine application (if not already created):
   ```
   gcloud app create --region=us-central1
   ```

6. Deploy the application:
   ```
   gcloud app deploy
   ```

7. View your deployed application:
   ```
   gcloud app browse
   ```

### Alternative Deployment Script

You can also use the provided deployment scripts:

- On Unix/Linux/MacOS: `./deploy.sh`
- On Windows: `deploy.bat`

## Project Structure

- `app.py`: Main Flask application
- `models/`: Deep learning model implementation
- `templates/`: HTML templates
- `uploads/`: Directory for uploaded data files
- `requirements.txt`: Python package dependencies
- `app.yaml`: Google App Engine configuration
- `main.py`: Entry point for Google App Engine

## Data Files

The application expects CSV files with exoplanet data. Sample files are included in the `uploads/` directory:
- kepler_df.csv: Kepler mission data
- toi_df.csv: TESS mission data
- k2_df.csv: K2 mission data

## Model Training

The application includes a deep learning model for exoplanet classification. The model is automatically trained when you upload data and click "Train Model" in the application interface.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
