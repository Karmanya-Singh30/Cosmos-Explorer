# Cosmos Explorer

Cosmos Explorer is an exoplanet detection application that uses machine learning to identify potential exoplanets from astronomical data.

## Features

- Upload and analyze exoplanet data from various sources (Kepler, TESS, K2)
- Train a deep learning model to classify exoplanet candidates
- Visualize exoplanet data and model predictions
- Interactive dashboard with detailed exoplanet information

## Prerequisites

- Python 3.9 or higher
- Google Cloud SDK (for deployment)
- Required Python packages (see requirements.txt)

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