# Deployment Guide for Cosmos Explorer

This guide explains how to deploy the Cosmos Explorer application to Google Cloud Platform.

## Deployment Options

There are several ways to deploy this application to Google Cloud:

1. Google App Engine (Standard or Flexible Environment)
2. Google Cloud Run (Container-based)
3. Google Kubernetes Engine (GKE)
4. Compute Engine (Virtual Machines)

## Google App Engine Deployment

### Prerequisites

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Create a Google Cloud Project (Project ID: `cosmos-explorer-474213`)
3. Enable billing for your project

### Steps

1. Install and initialize the Google Cloud SDK:
   ```bash
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```

3. Set your project ID:
   ```bash
   gcloud config set project cosmos-explorer-474213
   ```

4. Enable required APIs:
   ```bash
   gcloud services enable appengine.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

5. Create an App Engine application (if not already created):
   ```bash
   gcloud app create --region=us-central1
   ```

6. Deploy the application:
   ```bash
   gcloud app deploy
   ```

7. View your deployed application:
   ```bash
   gcloud app browse
   ```

## Google Cloud Run Deployment

### Prerequisites

1. Install Docker
2. Install the Google Cloud SDK
3. Enable the Cloud Run API

### Steps

1. Build the Docker image:
   ```bash
   docker build -t gcr.io/cosmos-explorer-474213/cosmos-explorer .
   ```

2. Push the image to Google Container Registry:
   ```bash
   docker push gcr.io/cosmos-explorer-474213/cosmos-explorer
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy cosmos-explorer \
       --image gcr.io/cosmos-explorer-474213/cosmos-explorer \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated
   ```

## Google Cloud Build Deployment

You can also use Google Cloud Build to automatically build and deploy your application:

1. Submit a build:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

## Environment Variables

The application supports the following environment variables:

- `FLASK_ENV`: Set to 'production' for production deployments
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (`cosmos-explorer-474213`)
- `MODEL_EPOCHS`: Number of training epochs (default: 50)
- `MODEL_BATCH_SIZE`: Training batch size (default: 32)
- `MODEL_LEARNING_RATE`: Learning rate for training (default: 0.001)

## Data Management

For production deployments, consider the following:

1. Use Google Cloud Storage for data files instead of local storage
2. Use Google Cloud SQL for any database needs
3. Implement proper logging with Google Cloud Logging
4. Set up monitoring with Google Cloud Monitoring

## Scaling Considerations

- For App Engine, configure automatic scaling in app.yaml
- For Cloud Run, configure concurrency and memory settings
- Consider using Google Cloud Memorystore for caching

## Security Considerations

- Use Google Cloud Secret Manager for sensitive configuration
- Implement proper authentication and authorization
- Use HTTPS for all communications
- Regularly update dependencies

## Troubleshooting

Common issues and solutions:

1. **Deployment fails due to missing dependencies**:
   - Ensure all dependencies are listed in requirements.txt
   - Check for compatibility with the runtime environment

2. **Application crashes on startup**:
   - Check the application logs: `gcloud app logs tail -s default`
   - Verify environment variables are set correctly

3. **Performance issues**:
   - Monitor resource usage in the Google Cloud Console
   - Consider optimizing the model or using more powerful instances

## Cost Management

- Monitor your usage in the Google Cloud Console
- Set up budget alerts to avoid unexpected charges
- Use preemptible instances for batch processing tasks
- Delete unused resources when not needed