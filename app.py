import os
from datetime import date, timedelta

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


# =========================================================
# Environment configuration
# =========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="AI-Powered Trip Planner",
    page_icon="✈️",
    layout="wide",
)

st.title("AI-Powered Trip Planner")
st.subheader("Plan Your Perfect Trip with CrewAI & OpenAI")

st.markdown(
    """
    Enter your trip details and preferences. A team of AI travel agents
    will create a personalized itinerary, budget, activity plan, and
    travel recommendations.
    """
)


# =========================================================
# Helper functions
# =========================================================
def calculate_trip_days(start_date: date, end_date: date) -> int:
    """Return the inclusive number of travel days."""
    return (end_date - start_date).days + 1


def create_trip_crew(
    origin_city: str,
    destination_city: str,
    start_date: date,
    end_date: date,
    interests: list[str],
    budget_range: str,
    travel_style: str,
) -> Crew:
    """
    Create and return a CrewAI crew for trip planning.
    """

    # CrewAI's own LLM wrapper.
    trip_llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.4,
    )

    trip_days = calculate_trip_days(start_date, end_date)
    interests_text = ", ".join(interests)

    # -----------------------------------------------------
    # Agents
    # -----------------------------------------------------
    destination_researcher = Agent(
        role="Destination Research Specialist",
        goal=(
            f"Research {destination_city} and identify the best attractions, "
            "neighborhoods, cultural experiences, practical travel advice, "
            "and destination-specific considerations."
        ),
        backstory=(
            "You are an experienced international travel researcher. "
            "You create accurate, practical, and destination-specific "
            "recommendations for travelers."
        ),
        llm=trip_llm,
        verbose=False,
        allow_delegation=False,
    )

    itinerary_designer = Agent(
        role="Travel Itinerary Designer",
        goal=(
            f"Create a realistic {trip_days}-day itinerary for "
            f"{destination_city} that matches the traveler's interests, "
            "budget, and preferred pace."
        ),
        backstory=(
            "You are a professional itinerary designer who organizes "
            "activities geographically and creates practical schedules "
            "without overloading the traveler."
        ),
        llm=trip_llm,
        verbose=False,
        allow_delegation=False,
    )

    budget_specialist = Agent(
        role="Travel Budget Specialist",
        goal=(
            "Develop a practical estimated travel budget covering "
            "transportation, lodging, meals, activities, and local travel."
        ),
        backstory=(
            "You are a travel-budget analyst who prepares transparent "
            "cost estimates and clearly identifies assumptions."
        ),
        llm=trip_llm,
        verbose=False,
        allow_delegation=False,
    )

    travel_editor = Agent(
        role="Senior Travel Plan Editor",
        goal=(
            "Combine all research, itinerary details, and budget estimates "
            "into one polished and easy-to-follow trip plan."
        ),
        backstory=(
            "You are a senior travel consultant who reviews plans for "
            "clarity, realism, safety, consistency, and usefulness."
        ),
        llm=trip_llm,
        verbose=False,
        allow_delegation=False,
    )

    # -----------------------------------------------------
    # Tasks
    # -----------------------------------------------------
    research_task = Task(
        description=f"""
Research a trip from {origin_city} to {destination_city}.

Travel information:
- Origin: {origin_city}
- Destination: {destination_city}
- Start date: {start_date.strftime("%B %d, %Y")}
- End date: {end_date.strftime("%B %d, %Y")}
- Duration: {trip_days} days
- Interests: {interests_text}
- Budget level: {budget_range}
- Travel style: {travel_style}

Provide:
1. A concise destination overview
2. Recommended neighborhoods or areas to stay
3. Major attractions related to the traveler's interests
4. Local cultural experiences
5. Dining recommendations and local specialties
6. Local transportation guidance
7. Safety, etiquette, weather, and practical travel tips
8. Important assumptions or details that should be verified

Do not claim live prices, availability, opening hours, or reservations
unless they are explicitly provided. Mark estimated information clearly.
""",
        expected_output=(
            "A structured destination research report with attractions, "
            "neighborhoods, food, transportation, and practical advice."
        ),
        agent=destination_researcher,
    )

    itinerary_task = Task(
        description=f"""
Using the destination research, create a complete day-by-day itinerary
for a {trip_days}-day trip to {destination_city}.

Traveler profile:
- Origin: {origin_city}
- Destination: {destination_city}
- Dates: {start_date.strftime("%B %d, %Y")} through
  {end_date.strftime("%B %d, %Y")}
- Interests: {interests_text}
- Budget: {budget_range}
- Travel style: {travel_style}

For every day, include:
- Morning plan
- Lunch suggestion
- Afternoon plan
- Dinner or evening suggestion
- Transportation guidance
- Estimated daily activity cost
- Optional alternative activity

Keep nearby activities together. Include arrival and departure
considerations where appropriate. Do not create impossible schedules.
""",
        expected_output=(
            f"A realistic {trip_days}-day itinerary organized by day, "
            "including meals, activities, transportation, and alternatives."
        ),
        agent=itinerary_designer,
        context=[research_task],
    )

    budget_task = Task(
        description=f"""
Create an estimated budget for this trip:

- Origin: {origin_city}
- Destination: {destination_city}
- Duration: {trip_days} days
- Budget level: {budget_range}
- Travel style: {travel_style}
- Interests: {interests_text}

Estimate the following categories:
1. Round-trip transportation
2. Lodging
3. Meals
4. Local transportation
5. Attractions and activities
6. Shopping or miscellaneous spending
7. Emergency contingency

Provide:
- Low estimate
- Expected estimate
- High estimate
- Estimated total per traveler
- Important assumptions

Clearly state that prices are estimates and should be verified before
booking. Do not present invented prices as confirmed live prices.
""",
        expected_output=(
            "A categorized trip-budget estimate with low, expected, and "
            "high totals plus assumptions."
        ),
        agent=budget_specialist,
        context=[research_task, itinerary_task],
    )

    final_plan_task = Task(
        description=f"""
Create the final personalized trip plan by combining the destination
research, itinerary, and budget.

Use this Markdown structure:

# {destination_city} Trip Plan

## Trip Summary
Include origin, destination, dates, duration, interests, budget level,
and travel style.

## Before You Go
Include practical preparation, documents, packing, transportation,
and reservation reminders.

## Recommended Area to Stay
Explain the recommended neighborhood or area and provide alternatives.

## Day-by-Day Itinerary
Give a detailed plan for every travel day.

## Estimated Budget
Provide the categorized budget table and total estimated range.

## Food and Dining Guide
Recommend local dishes, dining approaches, and neighborhood suggestions.

## Local Transportation
Explain how to move around efficiently.

## Safety and Cultural Tips
Provide practical, neutral guidance.

## Booking Checklist
List items the traveler should verify or reserve.

## Final Notes
Clearly state that prices, availability, schedules, visa rules, weather,
and opening hours should be independently verified before travel.

Make the final response polished, practical, and easy to read.
Do not mention internal agents or task execution.
""",
        expected_output=(
            "A complete, polished Markdown travel plan containing a trip "
            "summary, itinerary, budget, food guide, transportation, safety "
            "guidance, and booking checklist."
        ),
        agent=travel_editor,
        context=[research_task, itinerary_task, budget_task],
    )

    # Tasks run in order because the process is sequential.
    return Crew(
        agents=[
            destination_researcher,
            itinerary_designer,
            budget_specialist,
            travel_editor,
        ],
        tasks=[
            research_task,
            itinerary_task,
            budget_task,
            final_plan_task,
        ],
        process=Process.sequential,
        verbose=False,
    )


def generate_trip_plan(
    origin_city: str,
    destination_city: str,
    start_date: date,
    end_date: date,
    interests: list[str],
    budget_range: str,
    travel_style: str,
) -> str:
    """Run the CrewAI crew and return the final trip plan."""

    crew = create_trip_crew(
        origin_city=origin_city,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        interests=interests,
        budget_range=budget_range,
        travel_style=travel_style,
    )

    result = crew.kickoff()

    # CrewAI returns a CrewOutput object. Its raw property normally
    # contains the final task output.
    if hasattr(result, "raw") and result.raw:
        return str(result.raw)

    return str(result)


# =========================================================
# Session state
# =========================================================
if "trip_plan" not in st.session_state:
    st.session_state.trip_plan = ""


# =========================================================
# Input form
# =========================================================
st.divider()

left_column, right_column = st.columns(2)

with left_column:
    st.header("Trip Details")

    origin_city = st.text_input(
        "Origin City",
        value="New York",
        placeholder="Example: Los Angeles",
    )

    destination_city = st.text_input(
        "Destination City",
        value="Paris",
        placeholder="Example: Tokyo",
    )

    default_start = date.today() + timedelta(days=30)
    default_end = default_start + timedelta(days=5)

    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=date.today(),
    )

    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
    )

    if end_date >= start_date:
        trip_days = calculate_trip_days(start_date, end_date)
        st.info(f"Trip Duration: {trip_days} days")
    else:
        st.error("End date must be on or after the start date.")

with right_column:
    st.header("Preferences")

    interest_options = [
        "Culture & Museums",
        "Food & Dining",
        "History",
        "Architecture",
        "Nature & Outdoors",
        "Adventure",
        "Shopping",
        "Nightlife",
        "Beaches",
        "Family Activities",
        "Photography",
        "Relaxation",
    ]

    selected_interests = st.multiselect(
        "Your Interests",
        options=interest_options,
        default=["Culture & Museums", "Food & Dining"],
    )

    budget_range = st.select_slider(
        "Budget Range",
        options=[
            "Budget",
            "Moderate",
            "Comfortable",
            "Luxury",
        ],
        value="Moderate",
    )

    travel_style = st.radio(
        "Travel Style",
        options=["Relaxed", "Balanced", "Packed"],
        index=1,
        horizontal=True,
    )


# =========================================================
# Generate trip plan
# =========================================================
generate_button = st.button(
    "Generate AI Trip Plan",
    type="primary",
    use_container_width=True,
)

if generate_button:
    if not OPENAI_API_KEY:
        st.error(
            "OPENAI_API_KEY was not found. Add it to your `.env` file "
            "and restart the application."
        )

    elif not origin_city.strip():
        st.warning("Please enter an origin city.")

    elif not destination_city.strip():
        st.warning("Please enter a destination city.")

    elif end_date < start_date:
        st.warning("The end date must be on or after the start date.")

    elif not selected_interests:
        st.warning("Please select at least one travel interest.")

    else:
        try:
            with st.status(
                "AI agents are planning your trip...",
                expanded=True,
            ) as status:
                st.write("Researching the destination...")
                st.write("Designing the itinerary...")
                st.write("Estimating the travel budget...")
                st.write("Preparing the final trip plan...")

                trip_plan = generate_trip_plan(
                    origin_city=origin_city.strip(),
                    destination_city=destination_city.strip(),
                    start_date=start_date,
                    end_date=end_date,
                    interests=selected_interests,
                    budget_range=budget_range,
                    travel_style=travel_style,
                )

                st.session_state.trip_plan = trip_plan

                status.update(
                    label="Your personalized trip plan is ready!",
                    state="complete",
                    expanded=False,
                )

        except Exception as error:
            st.session_state.trip_plan = ""
            st.error(f"Error generating trip plan: {error}")


# =========================================================
# Display generated result
# =========================================================
if st.session_state.trip_plan:
    st.divider()
    st.header("Your Trip Plan")
    st.markdown(st.session_state.trip_plan)

    st.download_button(
        label="Download Trip Plan",
        data=st.session_state.trip_plan,
        file_name="ai_trip_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Trip Planner")

    st.write("**Framework**")
    st.code("CrewAI")

    st.write("**Model**")
    st.code("OpenAI GPT-4o mini")

    st.write("**Process**")
    st.code("Sequential multi-agent workflow")

    st.markdown("---")

    if OPENAI_API_KEY:
        st.success("OpenAI API key loaded")
    else:
        st.error("OpenAI API key missing")

    st.caption(
        "AI-generated recommendations and cost estimates should be "
        "verified before booking."
    )