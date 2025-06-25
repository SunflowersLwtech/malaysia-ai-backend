# Test Malaysia AI Backend Deployment
param(
    [Parameter(Mandatory=$true)]
    [string]$CloudRunURL
)

Write-Host "🧪 Testing Malaysia AI Backend Deployment" -ForegroundColor Cyan
Write-Host "URL: $CloudRunURL" -ForegroundColor Yellow
Write-Host "=" * 50

# Test 1: Health Check
Write-Host "1. Testing Health Endpoint..." -ForegroundColor Green
try {
    $healthResponse = Invoke-RestMethod -Uri "$CloudRunURL/health" -Method Get
    Write-Host "✅ Health Check: " -ForegroundColor Green -NoNewline
    Write-Host $healthResponse.status -ForegroundColor White
} catch {
    Write-Host "❌ Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Status Check
Write-Host "`n2. Testing Status Endpoint..." -ForegroundColor Green
try {
    $statusResponse = Invoke-RestMethod -Uri "$CloudRunURL/api/status" -Method Get
    Write-Host "✅ Backend Ready: " -ForegroundColor Green -NoNewline
    Write-Host $statusResponse.is_ready -ForegroundColor White
    Write-Host "✅ Version: " -ForegroundColor Green -NoNewline
    Write-Host $statusResponse.version -ForegroundColor White
} catch {
    Write-Host "❌ Status Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Chat Endpoint
Write-Host "`n3. Testing Chat Endpoint..." -ForegroundColor Green
try {
    $chatBody = @{
        message = "What are the best Malaysian dishes?"
        user_id = "test_user"
    } | ConvertTo-Json

    $chatResponse = Invoke-RestMethod -Uri "$CloudRunURL/api/chat" -Method Post -Body $chatBody -ContentType "application/json"
    Write-Host "✅ Chat Response: " -ForegroundColor Green -NoNewline
    Write-Host $chatResponse.message.Substring(0, [Math]::Min(100, $chatResponse.message.Length)) -ForegroundColor White
    Write-Host "✅ Success: " -ForegroundColor Green -NoNewline
    Write-Host $chatResponse.success -ForegroundColor White
} catch {
    Write-Host "❌ Chat Test Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Map Destinations
Write-Host "`n4. Testing Map Destinations..." -ForegroundColor Green
try {
    $mapResponse = Invoke-RestMethod -Uri "$CloudRunURL/api/map/destinations" -Method Get
    Write-Host "✅ Destinations Found: " -ForegroundColor Green -NoNewline
    Write-Host $mapResponse.total_count -ForegroundColor White
} catch {
    Write-Host "❌ Map Test Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n" + "=" * 50
Write-Host "🎉 Deployment Test Complete!" -ForegroundColor Cyan
Write-Host "Your Malaysia AI Backend is live at: $CloudRunURL" -ForegroundColor Yellow

# Usage instructions
Write-Host "`n📝 Usage:"
Write-Host "To run this test: .\test_deployment.ps1 -CloudRunURL 'https://your-cloud-run-url'" 