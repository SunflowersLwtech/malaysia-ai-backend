# Enhanced Frontend Features for Malaysia AI Travel Guide

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

class EnhancedMalaysiaAIFrontend:
    """Enhanced frontend features for better user experience"""
    
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url
        
    def create_chat_interface(self):
        """Enhanced chat interface with better UX"""
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .chat-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px 0;
            text-align: right;
        }
        .ai-message {
            background-color: #28a745;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px 0;
        }
        .sidebar-info {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar with quick actions
        with st.sidebar:
            st.header("🇲🇾 Malaysia Travel Assistant")
            
            # Quick suggestion buttons
            st.subheader("Quick Questions")
            if st.button("🏛️ KL Attractions"):
                st.session_state.quick_question = "What are the best attractions in Kuala Lumpur?"
            if st.button("🍜 Penang Food"):
                st.session_state.quick_question = "What are the must-try foods in Penang?"
            if st.button("🏝️ Island Recommendations"):
                st.session_state.quick_question = "Which Malaysian islands should I visit?"
            if st.button("💰 Budget Planning"):
                st.session_state.quick_question = "How much budget do I need for 5 days in Malaysia?"
                
            # Travel preferences
            st.subheader("Travel Preferences")
            budget_level = st.selectbox("Budget Level", ["Budget", "Mid-range", "Luxury"])
            interests = st.multiselect("Interests", 
                ["Food", "Culture", "Nature", "Shopping", "Adventure", "Relaxation"])
            duration = st.slider("Trip Duration (days)", 1, 30, 7)
            
            # Save preferences
            if st.button("Save Preferences"):
                st.session_state.user_preferences = {
                    "budget": budget_level,
                    "interests": interests,
                    "duration": duration
                }
                st.success("Preferences saved!")
        
        # Main chat interface
        st.title("🤖 TourismMalaysiaAI Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{message["content"]}</div>', 
                           unsafe_allow_html=True)
        
        # Handle quick questions
        if hasattr(st.session_state, 'quick_question'):
            user_input = st.session_state.quick_question
            delattr(st.session_state, 'quick_question')
        else:
            user_input = st.chat_input("Ask me anything about traveling in Malaysia...")
        
        # Process user input
        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            try:
                response = requests.post(
                    f"{self.api_base_url}/api/chat",
                    json={"message": user_input},
                    timeout=30
                )
                
                if response.status_code == 200:
                    ai_response = response.json()["response"]
                    sources = response.json().get("sources", [])
                    
                    # Add AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # Show sources if available
                    if sources:
                        with st.expander("📚 Knowledge Sources"):
                            for i, source in enumerate(sources[:3]):
                                st.write(f"**Source {i+1}:** {source.get('metadata', {}).get('category', 'General')}")
                                st.write(source['content'][:200] + "...")
                else:
                    st.error("Sorry, I'm having trouble connecting. Please try again.")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
            
            # Rerun to update the interface
            st.rerun()
    
    def create_itinerary_planner(self):
        """Interactive itinerary planning tool"""
        st.header("📅 Smart Itinerary Planner")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trip Details")
            duration = st.number_input("Duration (days)", min_value=1, max_value=30, value=7)
            budget_level = st.selectbox("Budget Level", ["budget", "mid-range", "luxury"])
            group_size = st.number_input("Group Size", min_value=1, max_value=20, value=2)
            starting_location = st.selectbox("Starting Location", 
                ["Kuala Lumpur", "Penang", "Johor Bahru", "Langkawi", "Kota Kinabalu"])
            
        with col2:
            st.subheader("Interests & Preferences")
            interests = st.multiselect("What interests you?", 
                ["food", "culture", "nature", "shopping", "adventure", "relaxation", "photography"])
            special_requirements = st.text_area("Special Requirements", 
                placeholder="Any dietary restrictions, accessibility needs, etc.")
        
        if st.button("🎯 Generate Itinerary", type="primary"):
            if interests:
                with st.spinner("Creating your personalized itinerary..."):
                    try:
                        response = requests.post(
                            f"{self.api_base_url}/api/generate-itinerary",
                            json={
                                "duration_days": duration,
                                "budget_level": budget_level,
                                "interests": interests,
                                "starting_location": starting_location,
                                "group_size": group_size,
                                "special_requirements": special_requirements
                            }
                        )
                        
                        if response.status_code == 200:
                            itinerary_data = response.json()
                            
                            st.success("🎉 Your itinerary is ready!")
                            st.markdown("### 📋 Your Malaysia Adventure Plan")
                            st.markdown(itinerary_data["itinerary"])
                            
                            # Download option
                            st.download_button(
                                "📥 Download Itinerary",
                                itinerary_data["itinerary"],
                                file_name=f"malaysia_itinerary_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("Unable to generate itinerary. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please select at least one interest!")
    
    def create_budget_calculator(self):
        """Interactive budget calculator"""
        st.header("💰 Budget Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=7)
            group_size = st.number_input("Number of People", min_value=1, max_value=20, value=2)
            budget_level = st.selectbox("Budget Style", ["budget", "mid-range", "luxury"])
            
        with col2:
            destinations = st.multiselect("Destinations to Visit", 
                ["Kuala Lumpur", "Penang", "Langkawi", "Cameron Highlands", "Malacca", 
                 "Johor Bahru", "Sabah", "Sarawak", "Ipoh", "Genting Highlands"])
        
        if st.button("💵 Calculate Budget"):
            if destinations:
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/budget-calculator",
                        json={
                            "duration_days": duration,
                            "destinations": destinations,
                            "budget_level": budget_level,
                            "group_size": group_size
                        }
                    )
                    
                    if response.status_code == 200:
                        budget_data = response.json()
                        
                        # Display budget breakdown
                        st.subheader("💸 Budget Breakdown")
                        
                        # Create budget visualization
                        categories = list(budget_data["budget_breakdown"].keys())
                        min_values = [budget_data["budget_breakdown"][cat]["min"] for cat in categories]
                        max_values = [budget_data["budget_breakdown"][cat]["max"] for cat in categories]
                        
                        # Budget chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='Minimum', x=categories, y=min_values))
                        fig.add_trace(go.Bar(name='Maximum', x=categories, y=max_values))
                        fig.update_layout(title='Budget Breakdown by Category (RM)', barmode='group')
                        st.plotly_chart(fig)
                        
                        # Summary cards
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Budget Range", 
                                     f"RM {budget_data['total_estimate']['min']:,.0f} - {budget_data['total_estimate']['max']:,.0f}")
                        with col2:
                            st.metric("Daily Average", 
                                     f"RM {budget_data['daily_average']['min']:,.0f} - {budget_data['daily_average']['max']:,.0f}")
                        with col3:
                            st.metric("Per Person", 
                                     f"RM {budget_data['per_person']['min']:,.0f} - {budget_data['per_person']['max']:,.0f}")
                        
                        # Money saving tips
                        st.subheader("💡 Money Saving Tips")
                        for tip in budget_data["money_saving_tips"]:
                            st.write(f"• {tip}")
                            
                except Exception as e:
                    st.error(f"Budget calculation error: {str(e)}")
            else:
                st.warning("Please select at least one destination!")
    
    def create_weather_widget(self):
        """Weather information widget"""
        st.header("🌤️ Weather Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.selectbox("Location", 
                ["Kuala Lumpur", "Penang", "Langkawi", "Cameron Highlands", "Sabah", "Sarawak"])
            month = st.selectbox("Month", 
                ["Any"] + [datetime(2024, i, 1).strftime("%B") for i in range(1, 13)])
        
        if st.button("🌡️ Get Weather Info"):
            try:
                month_num = None if month == "Any" else datetime.strptime(month, "%B").month
                
                response = requests.get(
                    f"{self.api_base_url}/api/weather/{location}",
                    params={"month": month_num} if month_num else {}
                )
                
                if response.status_code == 200:
                    weather_data = response.json()["weather"]
                    
                    st.subheader(f"🌍 {location} Weather Information")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Climate", weather_data["climate"].title())
                    with col2:
                        st.metric("Average Temperature", weather_data["avg_temp"])
                    with col3:
                        st.metric("Best Time to Visit", weather_data["best_time"])
                    
                    if "current_month" in weather_data:
                        month_info = weather_data["current_month"]
                        st.info(f"**{month}:** {month_info.get('description', 'No specific information')}")
                        
            except Exception as e:
                st.error(f"Weather information error: {str(e)}")
    
    def create_language_helper(self):
        """Language learning helper"""
        st.header("🗣️ Bahasa Malaysia Helper")
        
        phrase_type = st.selectbox("Phrase Category", 
            ["basic", "food", "directions", "emergency", "shopping"])
        
        if st.button("📖 Get Phrases"):
            try:
                response = requests.get(
                    f"{self.api_base_url}/api/language-guide",
                    params={"phrase_type": phrase_type}
                )
                
                if response.status_code == 200:
                    guide_data = response.json()
                    
                    st.subheader(f"📚 {phrase_type.title()} Phrases")
                    
                    # Display cultural tips
                    with st.expander("💡 Cultural Tips"):
                        for tip in guide_data["cultural_tips"]:
                            st.write(f"• {tip}")
                    
                    # Display phrases
                    phrases = guide_data["phrases"]
                    for category, phrase_dict in phrases.items():
                        st.write(f"**{category.title()}:**")
                        for english, malay in phrase_dict.items():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"🇬🇧 {english}")
                            with col2:
                                st.write(f"🇲🇾 {malay}")
                        st.write("---")
                        
            except Exception as e:
                st.error(f"Language guide error: {str(e)}")
    
    def create_dashboard(self):
        """Main dashboard with all features"""
        st.set_page_config(
            page_title="Malaysia AI Travel Guide",
            page_icon="🇲🇾",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Navigation
        page = st.sidebar.selectbox("Choose a feature", 
            ["💬 Chat Assistant", "📅 Itinerary Planner", "💰 Budget Calculator", 
             "🌤️ Weather Guide", "🗣️ Language Helper"])
        
        # Route to appropriate page
        if page == "💬 Chat Assistant":
            self.create_chat_interface()
        elif page == "📅 Itinerary Planner":
            self.create_itinerary_planner()
        elif page == "💰 Budget Calculator":
            self.create_budget_calculator()
        elif page == "🌤️ Weather Guide":
            self.create_weather_widget()
        elif page == "🗣️ Language Helper":
            self.create_language_helper()

# Usage example:
# frontend = EnhancedMalaysiaAIFrontend("https://your-api-url")
# frontend.create_dashboard()

# Additional frontend enhancements:

def create_interactive_map():
    """Create interactive Malaysia map with attractions"""
    malaysia_attractions = {
        "Kuala Lumpur": {"lat": 3.1390, "lon": 101.6869, "attractions": ["Petronas Towers", "Batu Caves", "KL Tower"]},
        "Penang": {"lat": 5.4164, "lon": 100.3327, "attractions": ["George Town", "Penang Hill", "Street Art"]},
        "Langkawi": {"lat": 6.3500, "lon": 99.8000, "attractions": ["Cable Car", "Underwater World", "Beaches"]},
        # Add more locations...
    }
    
    # Create map using Plotly
    fig = go.Figure(go.Scattermapbox(
        lat=[loc["lat"] for loc in malaysia_attractions.values()],
        lon=[loc["lon"] for loc in malaysia_attractions.values()],
        mode='markers',
        marker=go.scattermapbox.Marker(size=12, color='red'),
        text=list(malaysia_attractions.keys()),
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=go.layout.mapbox.Center(lat=4.5, lon=102.5), zoom=5),
        showlegend=False,
        height=600
    )
    
    return fig

def create_cultural_calendar():
    """Create interactive cultural events calendar"""
    events_data = {
        "January": ["Chinese New Year", "Thaipusam"],
        "February": ["Chinese New Year celebrations"],
        "May": ["Wesak Day", "Labour Day"],
        "August": ["Merdeka Day"],
        "November": ["Deepavali"],
        "December": ["Christmas"]
    }
    
    # Create calendar visualization
    calendar_df = pd.DataFrame([(month, event) for month, events in events_data.items() for event in events])
    calendar_df.columns = ["Month", "Event"]
    
    fig = px.timeline(calendar_df, x_start="Month", x_end="Month", y="Event", 
                     title="Malaysia Cultural Events Calendar")
    return fig 