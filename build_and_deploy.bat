@echo off
echo ========================================
echo Malaysia AI Travel Guide - Deployment
echo ========================================
echo.

echo Step 1: Building Docker image locally...
docker build -t malaysia-ai-guide .

if %ERRORLEVEL% neq 0 (
    echo ERROR: Docker build failed!
    echo Make sure Docker is installed and running.
    pause
    exit /b 1
)

echo.
echo ✅ Docker image built successfully!
echo.
echo Step 2: Tag image for Google Container Registry...
docker tag malaysia-ai-guide gcr.io/bright-coyote-463315-q8/malaysia-ai-guide

echo.
echo Step 3: Instructions for pushing to Google Cloud...
echo.
echo MANUAL STEPS (copy these commands):
echo.
echo 1. Configure Docker for GCR:
echo    gcloud auth configure-docker
echo.
echo 2. Push the image:
echo    docker push gcr.io/bright-coyote-463315-q8/malaysia-ai-guide
echo.
echo 3. Deploy to Cloud Run:
echo    gcloud run deploy malaysia-ai-guide --image gcr.io/bright-coyote-463315-q8/malaysia-ai-guide --platform managed --region us-west1 --allow-unauthenticated --memory 2Gi --cpu 2 --set-env-vars GEMINI_API_KEY=AIzaSyCS__n781EsrrX80XVVcTf2biRdMaftsK4
echo.
echo ========================================
echo Next: Copy the commands above and run them
echo ========================================
pause 