# 🚀 Quick Render Deployment Guide - Malaysia Tourism AI

## ⚡ **URGENT FIX: Authentication Error**

If you're seeing **"Your default credentials were not found"** error, follow these steps:

### 🔧 **Immediate Fix Steps:**

#### 1. **📋 MANUALLY Add Environment Variable in Render Dashboard**

**🔐 SECURITY NOTE:** For security reasons, the Google credentials CANNOT be stored in code files. You must add this manually in Render.

**Go to Render Dashboard → malaysia-ai-backend service → Environment → Add Environment Variable:**

**Variable Name:** `GOOGLE_SERVICE_ACCOUNT_JSON`

**Variable Value:** 
```
Copy the complete JSON content from your local file:
bright-coyote-463315-q8-59797318b374.json

⚠️ IMPORTANT: This JSON contains private keys - never store in code!
```

#### 2. **✅ Required Environment Variables:**
```bash
GOOGLE_CLOUD_PROJECT=bright-coyote-463315-q8
GOOGLE_CLOUD_LOCATION=us-west1  
VERTEX_AI_ENDPOINT=projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824
GOOGLE_SERVICE_ACCOUNT_JSON=[Your complete service account JSON - add manually in Render]
```

#### 3. **🔄 Restart Service:**
After adding the environment variable:
1. Go to Render Dashboard
2. Click "Manual Deploy" or wait for auto-deploy
3. Check logs for successful authentication

### 📊 **Expected Success Logs:**
```bash
✅ Google Gen AI client initialized successfully
✅ Using fine-tuned model endpoint: projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824
✅ Backend initialization complete
```

### 🌍 **Access URLs:**
- **Backend API:** https://malaysia-ai-backend.onrender.com
- **Frontend UI:** https://malaysia-ai-frontend.onrender.com
- **Health Check:** https://malaysia-ai-backend.onrender.com/health

### 🔍 **Troubleshooting:**
1. **JWT Signature Error:** Ensure service account JSON is correct
2. **Missing Variables:** Check all environment variables are set
3. **Cold Start:** First request after sleep may take 10-30 seconds

## 🎯 **Testing:**
Once deployed, test with Malaysia tourism questions:
- "Tell me about Kuala Lumpur attractions"
- "Best food in Malaysia"
- "Places to visit in Penang"

**🎉 Your Malaysia Tourism AI is now live on the cloud!** 