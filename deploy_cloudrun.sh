#!/bin/bash

# Malaysia AI Travel Guide - Cloud Run Deployment Script
# Deploy FastAPI backend to Google Cloud Run

set -e

# Configuration
PROJECT_ID="bright-coyote-463315-q8"
SERVICE_NAME="malaysia-ai-guide"
REGION="us-west1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Malaysia AI Travel Guide to Cloud Run"
echo "=================================================="
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Set active project
echo "📋 Setting Google Cloud project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push Docker image
echo "🏗️ Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 100 \
  --max-instances 10 \
  --set-env-vars GEMINI_API_KEY="AIzaSyCS__n781EsrrX80XVVcTf2biRdMaftsK4" \
  --port 8000

# Get the service URL
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo ""
echo "🌐 Your API is now live at:"
echo "   ${SERVICE_URL}"
echo ""
echo "📚 Test endpoints:"
echo "   Health: ${SERVICE_URL}/health"
echo "   Status: ${SERVICE_URL}/api/status"
echo "   Chat: ${SERVICE_URL}/api/chat"
echo "   Map: ${SERVICE_URL}/api/map/destinations"
echo ""
echo "🎉 Malaysia AI Travel Guide is ready on Cloud Run!" 