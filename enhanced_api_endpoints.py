# Enhanced API Endpoints for Malaysia AI Travel Guide

from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json

# Additional Pydantic models
class ItineraryRequest(BaseModel):
    duration_days: int
    budget_level: str  # "budget", "mid-range", "luxury"
    interests: List[str]  # ["food", "culture", "nature", "shopping", "adventure"]
    starting_location: str
    group_size: int = 2
    special_requirements: Optional[str] = None

class WeatherRequest(BaseModel):
    location: str
    month: Optional[int] = None

class BudgetRequest(BaseModel):
    duration_days: int
    destinations: List[str]
    budget_level: str
    group_size: int = 2

class RecommendationRequest(BaseModel):
    category: str  # "food", "destinations", "activities", "accommodation"
    location: Optional[str] = None
    budget_range: Optional[str] = None
    preferences: Optional[List[str]] = None

# New API endpoints to add to the main FastAPI app

@app.get("/api/weather/{location}")
async def get_weather_info(location: str, month: int = Query(None, ge=1, le=12)):
    """Get weather information for Malaysian locations"""
    try:
        weather_data = {
            "kuala lumpur": {
                "climate": "tropical",
                "avg_temp": "26-32°C",
                "rainy_season": "April-October",
                "dry_season": "November-March",
                "best_time": "December-February, May-July",
                "monthly_details": {
                    1: {"temp": "26-32°C", "rainfall": "Low", "description": "Cool and dry, excellent for tourism"},
                    2: {"temp": "27-33°C", "rainfall": "Low", "description": "Hot and dry, good for outdoor activities"},
                    3: {"temp": "27-33°C", "rainfall": "Medium", "description": "Getting warmer, occasional showers"},
                    4: {"temp": "28-33°C", "rainfall": "High", "description": "Rainy season begins, hot and humid"},
                    5: {"temp": "28-32°C", "rainfall": "Medium", "description": "Moderate rainfall, still good for travel"},
                    6: {"temp": "28-32°C", "rainfall": "Medium", "description": "Intermittent showers, pleasant evenings"},
                    7: {"temp": "27-32°C", "rainfall": "Medium", "description": "Dry spell, excellent weather"},
                    8: {"temp": "27-32°C", "rainfall": "Medium", "description": "Occasional showers, good for tourism"},
                    9: {"temp": "27-32°C", "rainfall": "High", "description": "Rainy season peak, plan indoor activities"},
                    10: {"temp": "27-32°C", "rainfall": "High", "description": "Heavy rainfall, monsoon season"},
                    11: {"temp": "26-31°C", "rainfall": "Medium", "description": "Transitioning to dry season"},
                    12: {"temp": "26-31°C", "rainfall": "Low", "description": "Cool and dry, peak tourism season"}
                }
            },
            "penang": {
                "climate": "tropical",
                "avg_temp": "25-31°C", 
                "rainy_season": "April-October",
                "dry_season": "November-March",
                "best_time": "December-April",
                "monthly_details": {
                    1: {"temp": "25-30°C", "rainfall": "Low", "description": "Perfect weather for George Town exploration"},
                    2: {"temp": "26-31°C", "rainfall": "Low", "description": "Ideal for beach activities"},
                    # ... similar structure for other months
                }
            }
            # Add more locations...
        }
        
        location_key = location.lower().replace(" ", " ")
        if location_key not in weather_data:
            return {"error": f"Weather data not available for {location}"}
        
        data = weather_data[location_key]
        if month:
            data["current_month"] = data["monthly_details"].get(month, {})
        
        return {"location": location, "weather": data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather data error: {str(e)}")

@app.post("/api/generate-itinerary")
async def generate_itinerary(request: ItineraryRequest):
    """Generate customized itinerary for Malaysia travel"""
    try:
        # Use the knowledge base and AI to generate itinerary
        context_query = f"{request.starting_location} {' '.join(request.interests)} {request.duration_days} days {request.budget_level}"
        knowledge_sources = search_knowledge_base(context_query, limit=10)
        context = "\n\n".join([source["content"] for source in knowledge_sources])
        
        itinerary_prompt = f"""Create a detailed {request.duration_days}-day Malaysia itinerary starting from {request.starting_location} for {request.group_size} people.

Budget Level: {request.budget_level}
Interests: {', '.join(request.interests)}
Special Requirements: {request.special_requirements or 'None'}

Provide day-by-day breakdown with:
- Specific attractions and activities
- Transportation details and costs
- Meal recommendations with prices
- Accommodation suggestions
- Daily budget estimates
- Cultural tips and local insights

Context from knowledge base:
{context}"""

        ai_response = generate_ai_response(itinerary_prompt, context, [])
        
        return {
            "itinerary": ai_response,
            "duration_days": request.duration_days,
            "budget_level": request.budget_level,
            "interests": request.interests,
            "sources": knowledge_sources[:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Itinerary generation error: {str(e)}")

@app.post("/api/budget-calculator")
async def calculate_budget(request: BudgetRequest):
    """Calculate estimated budget for Malaysia travel"""
    try:
        budget_ranges = {
            "budget": {
                "accommodation": (15, 60),  # RM per night
                "food": (15, 35),           # RM per day
                "transport": (10, 30),      # RM per day
                "activities": (0, 50),      # RM per day
                "miscellaneous": (10, 20)   # RM per day
            },
            "mid-range": {
                "accommodation": (80, 250),
                "food": (40, 80),
                "transport": (30, 80),
                "activities": (50, 150),
                "miscellaneous": (20, 50)
            },
            "luxury": {
                "accommodation": (300, 2000),
                "food": (100, 300),
                "transport": (80, 200),
                "activities": (150, 500),
                "miscellaneous": (50, 200)
            }
        }
        
        ranges = budget_ranges.get(request.budget_level, budget_ranges["mid-range"])
        
        # Calculate estimates
        accommodation_total = ranges["accommodation"][0] * request.duration_days * (request.group_size // 2 + request.group_size % 2)
        food_total = ranges["food"][0] * request.duration_days * request.group_size
        transport_total = ranges["transport"][0] * request.duration_days * request.group_size
        activities_total = ranges["activities"][0] * request.duration_days * request.group_size
        misc_total = ranges["miscellaneous"][0] * request.duration_days * request.group_size
        
        total_min = accommodation_total + food_total + transport_total + activities_total + misc_total
        
        # Maximum estimates
        accommodation_max = ranges["accommodation"][1] * request.duration_days * (request.group_size // 2 + request.group_size % 2)
        food_max = ranges["food"][1] * request.duration_days * request.group_size
        transport_max = ranges["transport"][1] * request.duration_days * request.group_size
        activities_max = ranges["activities"][1] * request.duration_days * request.group_size
        misc_max = ranges["miscellaneous"][1] * request.duration_days * request.group_size
        
        total_max = accommodation_max + food_max + transport_max + activities_max + misc_max
        
        return {
            "budget_breakdown": {
                "accommodation": {"min": accommodation_total, "max": accommodation_max},
                "food": {"min": food_total, "max": food_max},
                "transportation": {"min": transport_total, "max": transport_max},
                "activities": {"min": activities_total, "max": activities_max},
                "miscellaneous": {"min": misc_total, "max": misc_max}
            },
            "total_estimate": {
                "min": total_min,
                "max": total_max,
                "currency": "MYR (Malaysian Ringgit)"
            },
            "daily_average": {
                "min": total_min / request.duration_days,
                "max": total_max / request.duration_days
            },
            "per_person": {
                "min": total_min / request.group_size,
                "max": total_max / request.group_size
            },
            "money_saving_tips": [
                "Use public transportation and Grab for city travel",
                "Eat at hawker centers and local restaurants",
                "Book accommodations in advance for better rates",
                "Look for combo tickets for multiple attractions",
                "Shop at local markets for souvenirs",
                "Use Touch 'n Go card for transportation discounts"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget calculation error: {str(e)}")

@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations based on preferences"""
    try:
        # Search knowledge base based on category and preferences
        search_query = f"{request.category} {request.location or ''} {' '.join(request.preferences or [])}"
        sources = search_knowledge_base(search_query, limit=8)
        
        # Filter sources by category if specified
        filtered_sources = []
        for source in sources:
            if request.category.lower() in source.get("metadata", {}).get("category", "").lower():
                filtered_sources.append(source)
        
        if not filtered_sources:
            filtered_sources = sources[:5]  # Fallback to general results
        
        # Generate AI recommendations
        rec_prompt = f"""Provide personalized {request.category} recommendations for Malaysia travel.

Location preference: {request.location or 'Any location in Malaysia'}
Budget range: {request.budget_range or 'Any budget'}
Specific preferences: {', '.join(request.preferences) if request.preferences else 'General recommendations'}

For each recommendation include:
- Name and location
- Why it's special
- Price range
- Best time to visit/experience
- Practical tips

Use the knowledge base context below:
{chr(10).join([source["content"] for source in filtered_sources[:5]])}"""

        ai_response = generate_ai_response(rec_prompt, "", [])
        
        return {
            "category": request.category,
            "recommendations": ai_response,
            "sources": filtered_sources[:5],
            "filters_applied": {
                "location": request.location,
                "budget_range": request.budget_range,
                "preferences": request.preferences
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations error: {str(e)}")

@app.get("/api/nearby-attractions")
async def get_nearby_attractions(
    location: str = Query(..., description="Current location in Malaysia"),
    radius: int = Query(50, description="Search radius in kilometers"),
    category: str = Query("all", description="Category filter: food, culture, nature, shopping, all")
):
    """Get attractions near a specific location"""
    try:
        # Search for attractions near the specified location
        search_query = f"near {location} attractions {category} within {radius}km"
        sources = search_knowledge_base(search_query, limit=10)
        
        # Generate AI response with nearby recommendations
        nearby_prompt = f"""List attractions and points of interest near {location} within {radius}km radius.

Focus on {category} category attractions.

For each attraction provide:
- Name and exact location
- Distance from {location}
- Type of attraction
- Entry fees and opening hours
- Transportation options
- Why it's worth visiting
- Estimated time needed

Use context from knowledge base:
{chr(10).join([source["content"] for source in sources[:5]])}"""

        ai_response = generate_ai_response(nearby_prompt, "", [])
        
        return {
            "base_location": location,
            "search_radius": f"{radius}km",
            "category_filter": category,
            "nearby_attractions": ai_response,
            "sources": sources[:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nearby attractions error: {str(e)}")

@app.get("/api/language-guide")
async def get_language_guide(phrase_type: str = Query("basic", description="Type: basic, food, directions, emergency, shopping")):
    """Get useful Bahasa Malaysia phrases for travelers"""
    try:
        phrase_guides = {
            "basic": {
                "greetings": {
                    "Hello": "Helo / Selamat pagi (morning) / Selamat petang (afternoon)",
                    "Thank you": "Terima kasih",
                    "Please": "Tolong",
                    "Excuse me": "Maaf",
                    "Sorry": "Minta maaf",
                    "Yes": "Ya", 
                    "No": "Tidak",
                    "Do you speak English?": "Boleh cakap bahasa Inggeris?",
                    "I don't understand": "Saya tak faham",
                    "How much?": "Berapa harga?"
                }
            },
            "food": {
                "ordering": {
                    "I want to order": "Saya nak order",
                    "What do you recommend?": "Apa yang awak cadangkan?",
                    "Is this halal?": "Ni halal ke?",
                    "Not spicy": "Tak pedas",
                    "Very spicy": "Pedas sangat",
                    "The bill, please": "Bill, tolong",
                    "It's delicious": "Sedap",
                    "I'm vegetarian": "Saya vegetarian",
                    "No pork": "Tak nak babi",
                    "Water": "Air"
                }
            },
            "directions": {
                "navigation": {
                    "Where is...?": "Mana...?",
                    "How to go to...?": "Macam mana nak pergi...?",
                    "Turn left": "Belok kiri",
                    "Turn right": "Belok kanan", 
                    "Go straight": "Jalan lurus",
                    "Near": "Dekat",
                    "Far": "Jauh",
                    "Bus stop": "Perhentian bas",
                    "Train station": "Stesen keretapi",
                    "Airport": "Lapangan terbang"
                }
            },
            "emergency": {
                "help": {
                    "Help!": "Tolong!",
                    "Call police": "Panggil polis",
                    "Call ambulance": "Panggil ambulans",
                    "I'm lost": "Saya sesat",
                    "I need a doctor": "Saya perlukan doktor",
                    "Where is the hospital?": "Mana hospital?",
                    "I lost my passport": "Pasport saya hilang",
                    "Emergency": "Kecemasan",
                    "Fire!": "Api!",
                    "I need help": "Saya perlukan bantuan"
                }
            },
            "shopping": {
                "bargaining": {
                    "How much is this?": "Berapa harga ni?",
                    "Too expensive": "Mahal sangat",
                    "Can you give discount?": "Boleh bagi diskaun?",
                    "Final price?": "Harga final?",
                    "I'll take it": "Saya ambil",
                    "Do you accept credit card?": "Terima kad kredit?",
                    "Receipt": "Resit",
                    "Size": "Saiz",
                    "Color": "Warna",
                    "Quality": "Kualiti"
                }
            }
        }
        
        guide = phrase_guides.get(phrase_type, phrase_guides["basic"])
        
        return {
            "phrase_type": phrase_type,
            "language": "Bahasa Malaysia",
            "pronunciation_note": "Bahasa Malaysia is phonetic - pronounce words as they are written",
            "cultural_tips": [
                "Malaysians appreciate when visitors try to speak local language",
                "English is widely understood in tourist areas",
                "Use 'Pak' (uncle) or 'Kak' (sister) as polite terms for older people",
                "Pointing with index finger is considered rude - use thumb instead"
            ],
            "phrases": guide
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language guide error: {str(e)}")

@app.get("/api/cultural-calendar")
async def get_cultural_calendar(month: int = Query(None, ge=1, le=12), year: int = Query(None)):
    """Get Malaysian cultural events and festivals calendar"""
    try:
        current_year = year or datetime.now().year
        
        cultural_calendar = {
            1: ["Chinese New Year (varies)", "Thaipusam (varies)", "Federal Territory Day (1st)"],
            2: ["Chinese New Year celebrations continue"],
            3: ["Holi Festival (varies)"],
            4: ["Good Friday (varies)", "Easter (varies)"],
            5: ["Labour Day (1st)", "Wesak Day (varies)", "Harvest Festival Sabah (30-31)"],
            6: ["Gawai Festival Sarawak (1-2)", "Yang di-Pertuan Agong's Birthday (varies)"],
            7: ["School holidays period"],
            8: ["Merdeka Day preparations", "National Day (31st)"],
            9: ["Malaysia Day (16th)"],
            10: ["Deepavali preparations (varies)"],
            11: ["Deepavali (varies)"],
            12: ["Christmas (25th)", "School holidays begin"]
        }
        
        if month:
            events = cultural_calendar.get(month, [])
            month_name = datetime(current_year, month, 1).strftime("%B")
            return {
                "month": month_name,
                "year": current_year,
                "events": events,
                "cultural_note": f"Events in {month_name} often include both religious and cultural celebrations reflecting Malaysia's diversity"
            }
        else:
            return {
                "year": current_year,
                "annual_calendar": cultural_calendar,
                "note": "Dates for religious festivals vary each year based on lunar calendar"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cultural calendar error: {str(e)}")

@app.get("/api/search-enhanced")
async def enhanced_search(
    query: str = Query(..., description="Search query"),
    category: str = Query("all", description="Category filter"),
    location: str = Query("all", description="Location filter"),
    budget: str = Query("all", description="Budget filter"),
    limit: int = Query(5, ge=1, le=20, description="Number of results")
):
    """Enhanced search with filters"""
    try:
        # Combine query with filters
        enhanced_query = f"{query}"
        if category != "all":
            enhanced_query += f" {category}"
        if location != "all":
            enhanced_query += f" {location}"
        if budget != "all":
            enhanced_query += f" {budget}"
            
        # Search knowledge base
        sources = search_knowledge_base(enhanced_query, limit=limit*2)
        
        # Apply filters to metadata
        filtered_sources = []
        for source in sources:
            metadata = source.get("metadata", {})
            
            # Category filter
            if category != "all" and category.lower() not in metadata.get("category", "").lower():
                continue
                
            # Location filter  
            if location != "all" and location.lower() not in metadata.get("location", "").lower():
                continue
                
            # Budget filter (simplified)
            if budget != "all":
                price_range = metadata.get("price_range", "").lower()
                if budget == "budget" and "expensive" in price_range:
                    continue
                elif budget == "luxury" and ("budget" in price_range or "cheap" in price_range):
                    continue
                    
            filtered_sources.append(source)
            
            if len(filtered_sources) >= limit:
                break
        
        return {
            "query": query,
            "filters": {
                "category": category,
                "location": location, 
                "budget": budget
            },
            "results_count": len(filtered_sources),
            "results": filtered_sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced search error: {str(e)}") 