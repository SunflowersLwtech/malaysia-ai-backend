# Malaysia AI Travel Guide - Complete Deployment Guide

## 🚀 Overview

This deployment uses the **complete RAG-powered version** with all advanced features:

- ✅ **ChromaDB Vector Database** - For semantic search
- ✅ **Sentence Transformers** - For embedding generation  
- ✅ **Vertex AI Integration** - Your fine-tuned TourismMalaysiaAI model
- ✅ **Google Sheets Feedback** - User feedback collection
- ✅ **FastAPI + Gunicorn** - Production-ready web server
- ✅ **Memory Management** - Conversation history tracking

## 📋 Prerequisites

### 1. Install Google Cloud SDK

**Windows (PowerShell as Administrator):**
```powershell
# Download and install Google Cloud SDK
Invoke-WebRequest -Uri "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe" -OutFile "GoogleCloudSDKInstaller.exe"
.\GoogleCloudSDKInstaller.exe
```

**Or download manually:** https://cloud.google.com/sdk/docs/install

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project bright-coyote-463315-q8
```

## 🏗️ Deployment Architecture

```
Gunicorn Server (Port $PORT)
├── FastAPI Application (api_server.py)
├── ChromaDB Vector Database
├── Sentence Transformers (all-MiniLM-L6-v2)
├── Vertex AI Model (TourismMalaysiaAI)
├── Google Sheets Integration
└── Conversation Memory
```

## 📦 Deployment Commands

### Option 1: PowerShell Script (Recommended for Windows)

```powershell
# Make sure you're in the malaysia-ai-backend directory
cd malaysia-ai-backend

# Run the complete deployment script
.\deploy_complete.ps1
```

### Option 2: Manual Commands

```bash
# Build the container
gcloud builds submit --tag gcr.io/bright-coyote-463315-q8/malaysia-ai-guide .

# Deploy to Cloud Run
gcloud run deploy malaysia-ai-guide \
  --image gcr.io/bright-coyote-463315-q8/malaysia-ai-guide \
  --platform managed \
  --region us-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars="PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false" \
  --port 8000
```

## 🔧 Key Deployment Features

### Gunicorn Configuration
```bash
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api_server:app
```

- **Workers**: 1 (optimal for Cloud Run)
- **Threads**: 8 (handles concurrent requests)
- **Timeout**: 0 (no timeout for long AI responses)
- **Port**: Dynamic `$PORT` from Cloud Run

### Resource Allocation
- **Memory**: 2GB (for ChromaDB and embeddings)
- **CPU**: 1 vCPU (sufficient for single worker)
- **Timeout**: 300 seconds (5 minutes for startup)
- **Concurrency**: 80 requests per instance

## 🧪 Testing the Deployment

After successful deployment, test these endpoints:

### 1. Health Check
```bash
curl https://your-service-url/health
```

### 2. Status Check
```bash
curl https://your-service-url/api/status
```

### 3. Chat Endpoint
```bash
curl -X POST https://your-service-url/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Kuala Lumpur attractions"}'
```

### 4. Map Endpoint
```bash
curl https://your-service-url/api/map
```

## 📊 Monitoring and Logs

### View Recent Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=malaysia-ai-guide" --limit 50
```

### Monitor Service Health
```bash
gcloud run services describe malaysia-ai-guide --region=us-west1
```

## 🔄 Update Deployment

To update the service:

1. Make your code changes
2. Run the deployment script again:
   ```powershell
   .\deploy_complete.ps1
   ```

## 🐛 Troubleshooting

### Container Startup Issues

1. **Check logs**:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision" --limit 20
   ```

2. **Common Issues**:
   - **Memory exceeded**: Increase memory allocation
   - **Startup timeout**: Check initialization code
   - **Port binding**: Ensure Gunicorn uses `$PORT`

### Service Account Issues

Ensure your service account key is in the container:
```dockerfile
COPY bright-coyote-463315-q8-59797318b374.json .
```

### Dependencies Issues

If build fails, check `requirements_complete.txt`:
- ChromaDB version compatibility
- Sentence Transformers dependencies
- Google Cloud libraries

## 🎯 Expected Features

Once deployed, your service will have:

- **Vector Search**: Semantic similarity using embeddings
- **AI Responses**: Fine-tuned TourismMalaysiaAI model
- **Conversation Memory**: Multi-turn chat capability
- **Fallback Handling**: Graceful degradation if services fail
- **Interactive Map**: Malaysia tourism locations
- **Feedback Collection**: Google Sheets integration

## 📈 Performance Optimization

The complete version is optimized for:
- **Cold Start**: < 30 seconds with proper caching
- **Response Time**: < 5 seconds for AI responses
- **Throughput**: 80 concurrent requests per instance
- **Memory Usage**: Efficient ChromaDB and embedding management

---

## 🔗 Service Information

- **Project**: bright-coyote-463315-q8
- **Service**: malaysia-ai-guide  
- **Region**: us-west1
- **Expected URL**: https://malaysia-ai-guide-371826976402.us-west1.run.app
- **Model Endpoint**: 4352232060597829632 (TourismMalaysiaAI) 