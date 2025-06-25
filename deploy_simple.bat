@echo off
echo === Malaysia AI Travel Guide - Complete Deployment ===
echo.

set PROJECT_ID=bright-coyote-463315-q8
set SERVICE_NAME=malaysia-ai-guide
set REGION=us-west1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo Project ID: %PROJECT_ID%
echo Service Name: %SERVICE_NAME%
echo Region: %REGION%
echo Image: %IMAGE_NAME%
echo.

echo Checking authentication...
gcloud auth list --filter=status:ACTIVE --format="value(account)" > temp_auth.txt
set /p AUTH_CHECK=<temp_auth.txt
del temp_auth.txt

if "%AUTH_CHECK%"=="" (
    echo ERROR: Not authenticated with Google Cloud
    echo Please run: gcloud auth login
    pause
    exit /b 1
)

echo Authenticated as: %AUTH_CHECK%
echo.

echo Setting project...
gcloud config set project %PROJECT_ID%

echo.
echo Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

echo.
echo Building container with complete RAG functionality...
echo This includes:
echo   - ChromaDB vector database
echo   - Sentence Transformers embeddings
echo   - Vertex AI integration
echo   - Google Sheets feedback
echo   - Full FastAPI with Gunicorn
echo.

gcloud builds submit --tag %IMAGE_NAME% .

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
  --image %IMAGE_NAME% ^
  --platform managed ^
  --region %REGION% ^
  --allow-unauthenticated ^
  --memory 2Gi ^
  --cpu 1 ^
  --timeout 300 ^
  --concurrency 80 ^
  --max-instances 10 ^
  --set-env-vars="PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false" ^
  --port 8000

if %ERRORLEVEL% equ 0 (
    echo.
    echo Deployment successful!
    echo.
    echo Getting service URL...
    gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)" > temp_url.txt
    set /p SERVICE_URL=<temp_url.txt
    del temp_url.txt
    
    echo Service URL: %SERVICE_URL%
    echo.
    echo Test endpoints:
    echo   Health check: %SERVICE_URL%/health
    echo   Status: %SERVICE_URL%/api/status
    echo   Chat: %SERVICE_URL%/api/chat (POST)
    echo   Map: %SERVICE_URL%/api/map
    echo.
    echo Malaysia AI Travel Guide is ready on Cloud Run!
) else (
    echo Deployment failed!
    pause
    exit /b 1
)

echo.
pause 