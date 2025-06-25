#!/bin/bash

echo "=== Malaysia AI Travel Guide - Complete Deployment Script ==="
echo "This script will deploy the complete RAG-powered version to Google Cloud Run"
echo ""

# Configuration
PROJECT_ID="bright-coyote-463315-q8"
SERVICE_NAME="malaysia-ai-guide"
REGION="us-west1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"
echo "Image: $IMAGE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK not found!"
    echo "Please install Google Cloud SDK first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
echo "🔐 Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "❌ Not authenticated with Google Cloud"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "📋 Setting project..."
gcloud config set project $PROJECT_ID

# Enable necessary APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the container
echo "🏗️  Building container with complete RAG functionality..."
echo "This includes:"
echo "  - ChromaDB vector database"
echo "  - Sentence Transformers embeddings"
echo "  - Vertex AI integration"
echo "  - Google Sheets feedback"
echo "  - Full FastAPI with Gunicorn"
echo ""

gcloud builds submit --tag $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars="PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false" \
  --port 8000

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "Service URL:"
    gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
    echo ""
    echo "🧪 Test endpoints:"
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    echo "  Health check: ${SERVICE_URL}/health"
    echo "  Status: ${SERVICE_URL}/api/status"
    echo "  Chat: ${SERVICE_URL}/api/chat (POST)"
    echo "  Map: ${SERVICE_URL}/api/map"
    echo ""
    echo "📊 Monitor logs:"
    echo "  gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"
    echo ""
    echo "🔄 Update deployment:"
    echo "  ./deploy_complete.sh"
else
    echo "❌ Deployment failed!"
    exit 1
fi 