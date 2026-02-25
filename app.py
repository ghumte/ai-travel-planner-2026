# -----------------------------
# IMPORT REQUIRED LIBRARIES
# -----------------------------

import streamlit as st                    # UI framework
from crewai import Agent, Task, Crew      # CrewAI multi-agent framework
from langchain_openai import ChatOpenAI   # OpenAI LLM wrapper
from datetime import datetime, timedelta  # Date handling
import os                                 # Environment variable access


# -----------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------

# Set up the web app title, icon, and layout
st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state to store generated trip plan
st.session_state.setdefault("trip_plan", None)


# -----------------------------
# STEP 1: CREATE AI AGENTS
# -----------------------------

def create_agents(api_key: str):
    """
    Creates two AI agents:
    1. City Information Expert (research agent)
    2. Itinerary Planner (planning agent)
    """

    # Initialize the OpenAI LLM model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key
    )

    # Agent 1: Research Agent
    city_expert = Agent(
        role="City Information Expert",
        goal="Provide comprehensive information about cities including attractions, culture, and local tips",
        backstory="Experienced travel researcher with deep knowledge of cities worldwide.",
        verbose=True,              # Shows reasoning steps in console
        allow_delegation=False,    # Prevents this agent from delegating tasks
        llm=llm
    )

    # Agent 2: Itinerary Planning Agent
    itinerary_planner = Agent(
        role="Itinerary Planner",
        goal="Create detailed, personalized day-by-day travel itineraries based on user preferences",
        backstory="Professional travel planner with years of experience creating customized trips.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return city_expert, itinerary_planner


# -----------------------------
# STEP 2: CREATE TASKS FOR AGENTS
# -----------------------------

def create_tasks(city_expert, itinerary_planner, origin, destination,
                 start_date, end_date, interests, budget, travel_style):
    """
    Creates tasks that each agent will perform.
    Task 1: Research the city.
    Task 2: Create itinerary using research context.
    """

    # Calculate trip duration
    duration = (end_date - start_date).days + 1

    # Convert interests list into readable string
    interests_str = ", ".join(interests)

    # -----------------------------
    # TASK 1: Research Task
    # -----------------------------

    research_task = Task(
        description=f"""
Research {destination} and provide:
- Top 5 must-visit attractions related to: {interests_str}
- Best restaurants and local cuisine
- Cultural highlights and local customs
- Transportation options within the city
- Weather considerations and safety tips
Focus on {budget} budget and {travel_style} travel style. Be concise and practical.
""",
        agent=city_expert,
        expected_output="Structured city info with practical travel tips"
    )

    # -----------------------------
    # TASK 2: Itinerary Task
    # -----------------------------
    # Notice: It uses research_task as context

    itinerary_task = Task(
        description=f"""
Create a {duration}-day itinerary from {origin} to {destination}.
Dates: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}
Interests: {interests_str} | Budget: {budget} | Style: {travel_style}

For each day include:
- Morning (9 AM-12 PM), Afternoon (12-6 PM), Evening (6-10 PM)
- Estimated daily cost range
- Transportation tips
""",
        agent=itinerary_planner,
        expected_output="Day-by-day itinerary with activities, timings, and practical details",
        context=[research_task]   # This ensures planner uses research output
    )

    return [research_task, itinerary_task]


# -----------------------------
# STEP 3: EXECUTE CREW WORKFLOW
# -----------------------------

def generate_trip_plan(api_key, origin, destination,
                       start_date, end_date, interests,
                       budget, travel_style):
    """
    Orchestrates the full multi-agent workflow:
    1. Create agents
    2. Create tasks
    3. Run Crew
    4. Return final output
    """

    try:
        # Create AI agents
        city_expert, itinerary_planner = create_agents(api_key)

        # Create tasks
        tasks = create_tasks(
            city_expert, itinerary_planner,
            origin, destination,
            start_date, end_date,
            interests, budget, travel_style
        )

        # Create Crew and execute workflow
        result = Crew(
            agents=[city_expert, itinerary_planner],
            tasks=tasks,
            verbose=True
        ).kickoff()

        return str(result)

    except Exception as e:
        return f"Error generating trip plan: {e}"


# -----------------------------
# STREAMLIT SIDEBAR (API CONFIG)
# -----------------------------

with st.sidebar:
    st.header("Configuration")

    # Read API key from environment OR manual input
    api_key = os.getenv("OPENAI_API_KEY", "") or st.text_input(
        "OpenAI API Key (required)",
        type="password"
    )


# -----------------------------
# MAIN USER INTERFACE
# -----------------------------

st.title("AI-Powered Trip Planner")
st.markdown("### Plan Your Perfect Trip with CrewAI & OpenAI")

col1, col2 = st.columns(2)

# LEFT COLUMN: Trip Details
with col1:
    st.subheader("Trip Details")

    origin = st.text_input("Origin City", value="New York")
    destination = st.text_input("Destination City", value="Paris")

    start_date = st.date_input(
        "Start Date",
        value=datetime.now() + timedelta(days=7),
        min_value=datetime.now()
    )

    end_date = st.date_input(
        "End Date",
        value=datetime.now() + timedelta(days=12),
        min_value=start_date
    )

    st.info(f"Trip Duration: {(end_date - start_date).days + 1} days")

# RIGHT COLUMN: Preferences
with col2:
    st.subheader("Preferences")

    interests = st.multiselect(
        "Your Interests",
        ["Culture & Museums", "Food & Dining", "Adventure & Sports",
         "Nature & Parks", "Shopping", "Nightlife",
         "History", "Beach & Water Activities", "Art & Architecture"],
        default=["Culture & Museums", "Food & Dining"]
    )

    budget = st.select_slider(
        "Budget Range",
        options=["Budget", "Moderate", "Comfortable", "Luxury"],
        value="Moderate"
    )

    travel_style = st.radio(
        "Travel Style",
        ["Relaxed", "Balanced", "Packed"],
        index=1,
        horizontal=True
    )


# -----------------------------
# GENERATE BUTTON LOGIC
# -----------------------------

if st.button("Generate AI Trip Plan"):

    # Basic input validation
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not origin or not destination:
        st.error("Please enter both origin and destination.")
    elif not interests:
        st.error("Please select at least one interest.")
    else:
        with st.status("AI Agents are planning your trip..."):

            st.session_state.trip_plan = generate_trip_plan(
                api_key, origin, destination,
                start_date, end_date,
                interests, budget, travel_style
            )


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

if st.session_state.trip_plan:

    st.success("Your personalized trip plan is ready!")
    st.markdown("### Your Trip Plan")
    st.markdown(st.session_state.trip_plan)
