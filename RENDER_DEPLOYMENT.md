# 🚀 Deploy Malaysia Tourism AI to Render

Complete guide to deploy your Malaysia Tourism AI system to Render cloud platform.

## 📋 Prerequisites

- [Render Account](https://render.com) (free tier available)
- GitHub account with forked repositories
- Google Cloud credentials and API keys

## 🏗️ Step 1: Backend Deployment

### 1.1 Create Backend Service

1. **Go to Render Dashboard:**
   - Visit https://dashboard.render.com/
   - Click "New +" → "Web Service"

2. **Connect Repository:**
   - Connect your GitHub account
   - Select `malaysia-ai-backend` repository
   - Branch: `main`

3. **Configure Build Settings:**
   ```
   Name: malaysia-ai-backend
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn api_server_genai:app --host 0.0.0.0 --port $PORT
   ```

### 1.2 Set Environment Variables

Add these environment variables in Render:

```
GOOGLE_CLOUD_PROJECT=bright-coyote-463315-q8
GOOGLE_CLOUD_LOCATION=us-west1
VERTEX_AI_ENDPOINT=projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824
GEMINI_API_KEY=your_api_key_here
```

**Important:** Replace `your_api_key_here` with your actual Gemini API key.

### 1.3 Deploy

- Click "Create Web Service"
- Wait for deployment (5-10 minutes)
- Note your backend URL: `https://your-backend-name.onrender.com`

## 🖥️ Step 2: Frontend Deployment

### 2.1 Create Frontend Service

1. **Create Another Web Service:**
   - Click "New +" → "Web Service"
   - Select `malaysia-ai-frontend` repository

2. **Configure Build Settings:**
   ```
   Name: malaysia-ai-frontend
   Environment: Python 3
   Build Command: pip install -r streamlit_requirements.txt
   Start Command: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
   ```

### 2.2 Set Environment Variables

```
API_BASE_URL=https://your-backend-name.onrender.com
```

**Important:** Use the actual backend URL from Step 1.3.

### 2.3 Deploy

- Click "Create Web Service"
- Wait for deployment
- Your app will be available at: `https://your-frontend-name.onrender.com`

## ✅ Step 3: Verify Deployment

### 3.1 Test Backend

Visit: `https://your-backend-name.onrender.com/health`

Should return:
```json
{
  "status": "healthy",
  "message": "AI Chat Backend (Google Gen AI SDK) is running",
  "model_endpoint": "projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824"
}
```

### 3.2 Test Frontend

1. Visit your frontend URL
2. Check sidebar shows "✅ Backend: Connected"
3. Send a test message about Malaysia tourism
4. Verify AI responds correctly

## 🔧 Troubleshooting

### Common Issues

**1. Backend shows "Application startup failed"**
- Check environment variables are set correctly
- Verify Google Cloud API key is valid

**2. Frontend shows "❌ Backend: Disconnected"**
- Ensure `API_BASE_URL` points to correct backend URL
- Check backend is healthy at `/health` endpoint

**3. "Invalid JWT Signature" errors**
- Verify Google Cloud credentials
- Check if API key has correct permissions

### Logs

- Check Render service logs in dashboard
- Look for startup errors or runtime issues

## 💰 Cost Considerations

**Render Free Tier:**
- ✅ Perfect for development and testing
- ✅ 750 hours/month free
- ❌ Services sleep after 15 minutes of inactivity
- ❌ Cold start delays (30-60 seconds)

**Render Paid Plans:**
- 🚀 Always-on services (no sleep)
- 🚀 Faster performance
- 🚀 Custom domains
- 💰 Starting at $7/month per service

## 🎯 Next Steps

1. **Custom Domain:** Add your own domain in Render dashboard
2. **HTTPS:** Automatically provided by Render
3. **Monitoring:** Set up health checks and alerts
4. **Scaling:** Upgrade to paid plan for production use

## 🔗 Useful Links

- [Backend Repository](https://github.com/SunflowersLwtech/malaysia-ai-backend)
- [Frontend Repository](https://github.com/SunflowersLwtech/malaysia-ai-frontend)
- [Render Documentation](https://render.com/docs)
- [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)

## 🎉 Congratulations!

Your Malaysia Tourism AI is now deployed globally and accessible from anywhere in the world! 🌍 