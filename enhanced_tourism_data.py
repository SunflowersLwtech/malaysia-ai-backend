# Enhanced Malaysia Tourism Data for ChromaDB
# This will replace the basic data in api_server.py

MALAYSIA_TOURISM_DATA = [
    # === MAJOR DESTINATIONS ===
    {
        "id": "dest_kl_detailed",
        "content": "Kuala Lumpur, Malaysia's capital, offers iconic Petronas Twin Towers (452m tall, observation deck RM85), KL Tower (421m, sky deck RM105), Batu Caves (Hindu temple, 272 steps, free entry), Central Market (cultural handicrafts, 10am-10pm), Chinatown (Petaling Street, night market 6pm-12am), Bukit Bintang (shopping district, Pavilion, Lot 10), Merdeka Square (historical independence site), Sultan Abdul Samad Building (Moorish architecture), Bird Park (world's largest free-flight aviary, RM67 adult), and diverse food courts like Jalan Alor.",
        "category": "destination",
        "location": "Kuala Lumpur",
        "tags": ["urban", "shopping", "culture", "architecture", "nightlife"],
        "price_range": "RM50-200 per day",
        "best_time": "May-July, December-February"
    },
    
    {
        "id": "dest_penang_detailed", 
        "content": "Penang Island, UNESCO World Heritage George Town features colonial architecture, street art (Armenian Street, Lebuh Chulia), clan houses (Khoo Kongsi RM10, Cheah Kongsi), Penang Hill (funicular railway RM30 return, sunset views), Kek Lok Si Temple (largest Buddhist temple, free entry, pagoda RM2), Gurney Drive (food courts, shopping), Tropical Spice Garden (RM25 adult), and famous hawker centers like Chulia Street, Kimberley Street, and New Lane. Street food includes assam laksa, char kway teow, cendol.",
        "category": "destination", 
        "location": "Penang",
        "tags": ["heritage", "food", "culture", "art", "beach"],
        "price_range": "RM40-150 per day",
        "best_time": "December-April"
    },

    # === FOOD CULTURE ===
    {
        "id": "food_nasi_lemak_comprehensive",
        "content": "Nasi Lemak, Malaysia's national dish, consists of coconut rice cooked with pandan leaves, served with sambal (chili paste), fried anchovies (ikan bilis), roasted peanuts, cucumber slices, and hard-boiled egg. Popular additions include rendang beef (RM15-25), fried chicken (RM8-15), squid sambal (RM12-18). Best places: Village Park Restaurant KL (famous fried chicken version), Nasi Lemak Wanjo (traditional style), Madam Kwan's (upscale version RM18-28). Street stalls typically RM3-8, restaurants RM10-25.",
        "category": "food",
        "location": "Malaysia",
        "tags": ["national dish", "breakfast", "coconut rice", "spicy"],
        "price_range": "RM3-28",
        "halal": True
    },

    # === ISLANDS & BEACHES ===
    {
        "id": "dest_langkawi_complete",
        "content": "Langkawi Archipelago (99 islands, duty-free status) offers Langkawi Cable Car (RM55 adult to Sky Bridge), Underwater World (RM38 adult), Eagle Square, Temurun Waterfall, Black Sand Beach (Pantai Pasir Tengkorak), Tanjung Rhu Beach (white sand, mangroves), Datai Bay (luxury resorts), island hopping tours (RM35-45 per person, includes Pulau Dayang Bunting/Pregnant Maiden Lake, Eagle Watching, Beras Basah Island). Duty-free shopping at Kuah town. Best accommodation: Four Seasons, Datai, Casa del Mar.",
        "category": "destination",
        "location": "Langkawi",
        "tags": ["island", "beach", "cable car", "duty-free", "nature"],
        "price_range": "RM100-500 per day",
        "best_time": "November-April"
    },

    # === CULTURAL EXPERIENCES ===
    {
        "id": "culture_festivals_detailed",
        "content": "Malaysia celebrates diverse festivals: Chinese New Year (January/February, 15 days, lion dances, ang pau, reunion dinners), Hari Raya Aidilfitri (end of Ramadan, open houses, ketupat, rendang), Deepavali (Festival of Lights, oil lamps, rangoli, murukku sweets), Christmas (December, decorations in malls), Wesak Day (Buddha's birthday, temple visits, free vegetarian meals), Thaipusam (Hindu festival, Batu Caves procession, kavadi carrying), Mid-Autumn Festival (mooncakes, lanterns). Many are public holidays with special promotions and cultural performances.",
        "category": "culture",
        "location": "Malaysia",
        "tags": ["festivals", "multicultural", "religious", "celebration"],
        "price_range": "Free-RM50 for special events",
        "best_time": "Year-round"
    },

    # === NATURE & ADVENTURE ===
    {
        "id": "dest_cameron_highlands",
        "content": "Cameron Highlands, hill station at 1500m elevation, offers cool climate (15-25°C), tea plantations (BOH Tea Plantation tours RM5, Bharat Tea Estate), strawberry farms (RM20-30 strawberry picking), butterfly gardens, Mossy Forest (guided tours RM25), Gunung Brinchang (highest peak, RM15 4WD tour), night markets (Brinchang, Tanah Rata), steamboat restaurants, and colonial-era accommodations. Popular for weekend getaways from KL (3-4 hours drive). Best for hiking, tea tasting, and escaping tropical heat.",
        "category": "destination",
        "location": "Cameron Highlands",
        "tags": ["highland", "tea", "cool climate", "nature", "hiking"],
        "price_range": "RM80-250 per day",
        "best_time": "March-September"
    },

    # === TRANSPORTATION ===
    {
        "id": "transport_comprehensive",
        "content": "Malaysia transportation: KLIA Express (airport to KL Sentral, RM55, 28 minutes), KTM trains (intercity, KL-Penang RM79-199), ETS (Electric Train Service, KL-Ipoh RM35-75), Grab (ride-hailing, widely available, RM10-50 city trips), local buses (RapidKL RM1-6, Touch 'n Go card), MRT/LRT (Kuala Lumpur metro, RM1-6), domestic flights (AirAsia, Malaysia Airlines, KL-Penang RM150-400), long-distance buses (KL-Singapore RM35-60), car rental (RM80-200/day, international license required), ferries (Penang-Langkawi RM70-120).",
        "category": "transportation",
        "location": "Malaysia", 
        "tags": ["public transport", "flights", "trains", "buses", "grab"],
        "price_range": "RM1-400",
        "best_time": "Year-round"
    },

    # === ACCOMMODATION ===
    {
        "id": "accommodation_guide",
        "content": "Malaysia accommodation ranges from budget hostels (RM25-60/night, dorms RM15-35), mid-range hotels (RM80-250/night), luxury resorts (RM300-2000/night). Popular areas: KL (Bukit Bintang, KLCC, KL Sentral), Penang (George Town heritage area, Batu Ferringhi beach), Langkawi (Pantai Cenang, Datai Bay), Malacca (Jonker Street area). Booking platforms: Agoda, Booking.com, local deals. Airbnb available in major cities. Resort islands offer all-inclusive packages. Check-in usually 3pm, check-out 12pm. Many include breakfast and WiFi.",
        "category": "accommodation",
        "location": "Malaysia",
        "tags": ["hotels", "hostels", "resorts", "booking", "budget"],
        "price_range": "RM15-2000 per night",
        "best_time": "Book advance for peak seasons"
    },

    # === SHOPPING ===
    {
        "id": "shopping_comprehensive", 
        "content": "Malaysia shopping: Kuala Lumpur malls (Pavilion KL luxury brands, Suria KLCC international, Mid Valley megamall, 1 Utama largest), Penang (Gurney Plaza, Queensbay Mall, heritage shophouses), Johor Bahru (Johor Premium Outlets duty-free). Local markets: Central Market KL (handicrafts), Petaling Street (replica goods), Night Markets (pasar malam, local snacks). Products: batik textiles, pewter items (Royal Selangor), tropical fruits, palm oil products, traditional medicines, electronics (Lowyat Plaza). Bargaining common in markets, fixed prices in malls. Sales periods: Year-end, Chinese New Year, Hari Raya.",
        "category": "shopping",
        "location": "Malaysia",
        "tags": ["malls", "markets", "handicrafts", "electronics", "bargaining"],
        "price_range": "RM5-5000+",
        "best_time": "Sale seasons, avoid weekends for crowds"
    },

    # === PRACTICAL INFO ===
    {
        "id": "practical_travel_info",
        "content": "Malaysia practical info: Currency Malaysian Ringgit (RM/MYR), exchange rate ~RM4.7=USD1. Languages: Bahasa Malaysia (official), English widely spoken, Chinese dialects, Tamil. Climate: tropical (26-32°C), rainy season varies by region. Visa: 90-day free entry for most countries. SIM cards available at airports (Maxis, Celcom, Digi, RM30-50 with data). Tipping not mandatory but appreciated (10% restaurants, RM2-5 guides). Prayer times observed, halal food widely available. Dress modestly at religious sites. Emergency numbers: 999 (police/ambulance), 994 (fire). Tourist police available in major areas.",
        "category": "practical",
        "location": "Malaysia", 
        "tags": ["currency", "language", "visa", "climate", "emergency"],
        "price_range": "Varies",
        "best_time": "Year-round destination"
    },

    # === REGIONAL SPECIALTIES ===
    {
        "id": "regional_foods",
        "content": "Regional Malaysian specialties: Penang (Assam Laksa RM6-12, Char Kway Teow RM7-15, Cendol RM3-6), Ipoh (White Coffee RM3-8, Bean Sprout Chicken RM15-25, Salted Chicken RM20-30), Johor (Laksa Johor RM8-15, Otak-otak RM1-3 per piece), Sabah (Hinava raw fish RM15-25, Tuaran Mee RM8-12), Sarawak (Sarawak Laksa RM6-10, Kolo Mee RM5-8), Kelantan (Nasi Kerabu blue rice RM8-15, Ayam Percik RM12-20), Malacca (Chicken Rice Balls RM8-15, Satay Celup RM25-35), Terengganu (Keropok Lekor RM5-10, Nasi Dagang RM8-12). Each state has unique flavors and cooking styles.",
        "category": "food",
        "location": "Malaysia",
        "tags": ["regional", "specialties", "local cuisine", "states"],
        "price_range": "RM3-35",
        "best_time": "Available year-round"
    }
]

# Categories for better organization
CATEGORIES = {
    "destination": "Tourist destinations and attractions",
    "food": "Malaysian cuisine and dining",
    "culture": "Cultural experiences and festivals", 
    "transportation": "Getting around Malaysia",
    "accommodation": "Places to stay",
    "shopping": "Shopping areas and products",
    "practical": "Travel tips and practical information"
}

# Enhanced metadata for better search
def get_enhanced_metadata(item):
    """Generate enhanced metadata for ChromaDB"""
    return {
        "category": item["category"],
        "location": item["location"],
        "tags": ",".join(item.get("tags", [])),
        "price_range": item.get("price_range", ""),
        "best_time": item.get("best_time", ""),
        "halal": item.get("halal", False),
        "id": item["id"]
    } 