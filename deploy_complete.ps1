# Malaysia AI Travel Guide - Complete Deployment Script (PowerShell)
Write-Host "=== Malaysia AI Travel Guide - Complete Deployment Script ===" -ForegroundColor Green
Write-Host "This script will deploy the complete RAG-powered version to Google Cloud Run" -ForegroundColor Yellow
Write-Host ""

# Configuration
$PROJECT_ID = "bright-coyote-463315-q8"
$SERVICE_NAME = "malaysia-ai-guide"
$REGION = "us-west1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "Project ID: $PROJECT_ID" -ForegroundColor Cyan
Write-Host "Service Name: $SERVICE_NAME" -ForegroundColor Cyan
Write-Host "Region: $REGION" -ForegroundColor Cyan
Write-Host "Image: $IMAGE_NAME" -ForegroundColor Cyan
Write-Host ""

# Check if gcloud is installed
try {
    $null = Get-Command gcloud -ErrorAction Stop
    Write-Host "✅ Google Cloud SDK found" -ForegroundColor Green
} catch {
    Write-Host "❌ Google Cloud SDK not found!" -ForegroundColor Red
    Write-Host "Please install Google Cloud SDK first:" -ForegroundColor Yellow
    Write-Host "https://cloud.google.com/sdk/docs/install" -ForegroundColor Blue
    exit 1
}

# Check authentication
Write-Host "🔐 Checking authentication..." -ForegroundColor Blue
$authCheck = gcloud auth list --filter=status:ACTIVE --format="value(account)"
if ([string]::IsNullOrEmpty($authCheck) -or $authCheck -notmatch "@") {
    Write-Host "❌ Not authenticated with Google Cloud" -ForegroundColor Red
    Write-Host "Please run: gcloud auth login" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Authenticated as: $authCheck" -ForegroundColor Green

# Set project
Write-Host "📋 Setting project..." -ForegroundColor Blue
gcloud config set project $PROJECT_ID

# Enable necessary APIs
Write-Host "🔧 Enabling required APIs..." -ForegroundColor Blue
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the container
Write-Host "🏗️  Building container with complete RAG functionality..." -ForegroundColor Magenta
Write-Host "This includes:" -ForegroundColor Yellow
Write-Host "  - ChromaDB vector database" -ForegroundColor White
Write-Host "  - Sentence Transformers embeddings" -ForegroundColor White
Write-Host "  - Vertex AI integration" -ForegroundColor White
Write-Host "  - Google Sheets feedback" -ForegroundColor White
Write-Host "  - Full FastAPI with Gunicorn" -ForegroundColor White
Write-Host ""

gcloud builds submit --tag $IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

# Deploy to Cloud Run
Write-Host "🚀 Deploying to Cloud Run..." -ForegroundColor Green
gcloud run deploy $SERVICE_NAME `
  --image $IMAGE_NAME `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --timeout 300 `
  --concurrency 80 `
  --max-instances 10 `
  --set-env-vars="PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false" `
  --port 8000

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Deployment successful!" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Service URL:" -ForegroundColor Cyan
    $SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
    Write-Host $SERVICE_URL -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "🧪 Test endpoints:" -ForegroundColor Yellow
    Write-Host "  Health check: $SERVICE_URL/health" -ForegroundColor White
    Write-Host "  Status: $SERVICE_URL/api/status" -ForegroundColor White
    Write-Host "  Chat: $SERVICE_URL/api/chat (POST)" -ForegroundColor White
    Write-Host "  Map: $SERVICE_URL/api/map" -ForegroundColor White
    Write-Host ""
    
    Write-Host "📊 Monitor logs:" -ForegroundColor Cyan
    Write-Host "  gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit 50" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "🔄 Update deployment:" -ForegroundColor Magenta
    Write-Host "  .\deploy_complete.ps1" -ForegroundColor Gray
    
} else {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
} 