# Cosmos Explorer Deployment Checklist

## Project Information
- **Project ID**: `cosmos-explorer-474213`
- **Region**: `us-central1` (default)

## Pre-deployment Checklist

### 1. Google Cloud SDK Setup
- [ ] Install Google Cloud SDK ([Download](https://cloud.google.com/sdk/docs/install))
- [ ] Initialize gcloud: `gcloud init`
- [ ] Authenticate: `gcloud auth login`

### 2. Project Configuration
- [ ] Set project: `gcloud config set project cosmos-explorer-474213`
- [ ] Verify project: `gcloud config list project`

### 3. Enable Required APIs
- [ ] App Engine Admin API: `gcloud services enable appengine.googleapis.com`
- [ ] Cloud Build API: `gcloud services enable cloudbuild.googleapis.com`
- [ ] Container Registry API: `gcloud services enable containerregistry.googleapis.com`

### 4. App Engine Setup
- [ ] Create App Engine app: `gcloud app create --region=us-central1`

### 5. File Preparation
- [ ] Verify all required files are present (run `python check_deployment.py`)
- [ ] Ensure [app.yaml](file://c:\Users\Karmanya\NASA%20H\app.yaml) is configured correctly
- [ ] Check that [requirements.txt](file://c:\Users\Karmanya\NASA%20H\requirements.txt) contains all dependencies

## Deployment Options

### Option 1: Google App Engine (Recommended)
```bash
gcloud app deploy
```

### Option 2: Google Cloud Run
```bash
# Build and push Docker image
docker build -t gcr.io/cosmos-explorer-474213/cosmos-explorer .
docker push gcr.io/cosmos-explorer-474213/cosmos-explorer

# Deploy to Cloud Run
gcloud run deploy cosmos-explorer \
    --image gcr.io/cosmos-explorer-474213/cosmos-explorer \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Option 3: Using Deployment Scripts
- **Windows**: Run `deploy_to_cosmos_explorer.bat`
- **Unix/Linux/Mac**: Run `deploy_to_cosmos_explorer.sh`

## Post-deployment Verification

### 1. Check Deployment Status
- [ ] View deployment logs: `gcloud app logs tail -s default`
- [ ] Check service status: `gcloud app services list`

### 2. Access Application
- [ ] Open in browser: `gcloud app browse`
- [ ] Or visit: `https://cosmos-explorer-474213.appspot.com`

### 3. Verify Functionality
- [ ] Test health check endpoint: `curl https://cosmos-explorer-474213.appspot.com/health`
- [ ] Verify main page loads correctly
- [ ] Test file upload functionality

## Troubleshooting

### Common Issues

1. **Permission denied errors**
   - Ensure you have the necessary IAM permissions
   - Check that billing is enabled for the project

2. **API not enabled**
   - Enable required APIs: `gcloud services enable appengine.googleapis.com cloudbuild.googleapis.com`

3. **Deployment timeouts**
   - Check logs: `gcloud app logs tail -s default`
   - Verify all dependencies are in [requirements.txt](file://c:\Users\Karmanya\NASA%20H\requirements.txt)

4. **Application crashes**
   - Check application logs: `gcloud app logs tail -s default`
   - Verify environment variables are set correctly

### Useful Commands

- **View logs**: `gcloud app logs tail -s default`
- **List services**: `gcloud app services list`
- **View versions**: `gcloud app versions list`
- **Open browser**: `gcloud app browse`
- **Check project**: `gcloud config list project`
- **List accounts**: `gcloud auth list`

## Support Resources

- [Google App Engine Documentation](https://cloud.google.com/appengine/docs)
- [Google Cloud Pricing](https://cloud.google.com/pricing)
- [Google Cloud Status](https://status.cloud.google.com/)
- [Google Cloud Support](https://cloud.google.com/support)

## Notes

- Deployment may take several minutes to complete
- First deployment will be slower than subsequent ones
- Make sure your Google Cloud account has billing enabled
- Keep your project ID (`cosmos-explorer-474213`) handy for all commands