# 🇲🇾 Malaysia Tourism AI Chatbot

A sophisticated AI-powered tourism assistant specializing in Malaysia travel recommendations, powered by fine-tuned Google Gemini AI.

## ✨ Features

- 🧠 **Fine-tuned Gemini AI** - Specialized knowledge about Malaysia tourism
- 🌍 **Complete Travel Guide** - Covers all Malaysian states and attractions
- 📱 **Mobile-Friendly** - Responsive Streamlit interface
- ⚡ **Real-time Responses** - Instant AI-powered recommendations
- 🚀 **Cloud Deployed** - Accessible globally via Render

## 🏗️ Architecture

```
Streamlit Frontend ↔ FastAPI Backend ↔ Fine-tuned Gemini Model
```

## 🚀 Quick Deploy to Render

### Backend Deployment

1. **Fork this repository**
2. **Connect to Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `malaysia-ai-backend` folder

3. **Configure Build Settings:**
   ```
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn api_server_genai:app --host 0.0.0.0 --port $PORT
   ```

4. **Set Environment Variables:**
   - `GOOGLE_CLOUD_PROJECT`: bright-coyote-463315-q8
   - `GOOGLE_CLOUD_LOCATION`: us-west1
   - `VERTEX_AI_ENDPOINT`: projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824
   - `GEMINI_API_KEY`: [Your Gemini API Key]

### Frontend Deployment

1. **Deploy Frontend:**
   - Create another Render Web Service
   - Point to same repository
   - Build Command: `pip install -r streamlit_requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

2. **Set Frontend Environment:**
   - `API_BASE_URL`: https://your-backend-url.onrender.com

## 🎯 Local Development

```bash
# Clone the repository
git clone https://github.com/SunflowersLwtech/malaysia-ai-backend.git
cd malaysia-ai-backend

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn api_server_genai:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend
pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

## 🔧 Configuration

Create a `.env` file with:
```
GOOGLE_CLOUD_PROJECT=bright-coyote-463315-q8
GOOGLE_CLOUD_LOCATION=us-west1
VERTEX_AI_ENDPOINT=projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824
GEMINI_API_KEY=your_api_key_here
```

## 🌟 API Endpoints

- `GET /health` - Health check
- `POST /chat` - Send message to AI
- `GET /docs` - API documentation

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use for your own projects!
