# app.py
import streamlit as st
import sys

# Workaround for sqlite3 version issue
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Set up the page
st.title("Radio Jingle Generator using CrewAI")
st.markdown("""
This app uses CrewAI to generate short radio jingles. It includes:
- **Researcher AI**: Researches the theme or product.
- **Creator AI**: Creates the initial jingle lyrics and structure.
- **Copywriter AI**: Polishes and refines the content for radio.

Enter a theme (e.g., "coffee shop promotion") and generate!
""")

# API Key handling
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
    st.stop()

# Check pip version
import pip
pip_version = pip.__version__
if pip_version < "25.2":
    st.warning(f"Using pip {pip_version}. Consider updating to 25.2 or later for better dependency resolution.")

# LLM setup
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.7
    )
except Exception as e:
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

# User input
theme = st.text_input("Enter the theme or product for the radio jingle:", "Example: Summer Beach Party")
generate_button = st.button("Generate Jingle")

if generate_button and theme:
    with st.spinner("Assembling the crew and generating your jingle..."):
        try:
            # Define Agents
            researcher = Agent(
                role="Researcher",
                goal="Research key facts, trends, and appealing elements about the theme to inspire the jingle.",
                backstory="You are an expert researcher with access to vast knowledge on various topics. Focus on fun, engaging, and relevant info for radio ads.",
                verbose=True,
                llm=llm
            )

            creator = Agent(
                role="Jingle Creator",
                goal="Create a short, catchy radio jingle based on research, including lyrics and simple structure suggestions.",
                backstory="You are a creative genius specializing in audio ads. Make it rhythmic, memorable, and under 30 seconds worth of content.",
                verbose=True,
                llm=llm
            )

            copywriter = Agent(
                role="Copywriter",
                goal="Refine and polish the jingle content for clarity, impact, and radio-friendliness.",
                backstory="You are a professional copywriter with experience in advertising. Ensure it's persuasive, error-free, and optimized for spoken delivery.",
                verbose=True,
                llm=llm
            )

            # Define Tasks
            research_task = Task(
                description=f"Research the theme: '{theme}'. Gather 5-7 key points that could make a jingle engaging.",
                expected_output="A bullet-point list of research findings.",
                agent=researcher
            )

            create_task = Task(
                description=f"Using the research, create a short radio jingle for '{theme}'. Include lyrics and notes on rhythm/timing.",
                expected_output="The jingle lyrics with structure (e.g., verse, chorus). Keep it concise.",
                agent=creator,
                context=[research_task]
            )

            copywrite_task = Task(
                description=f"Refine the created jingle for '{theme}'. Improve flow, add punch, ensure it's radio-ready.",
                expected_output="The final polished jingle script.",
                agent=copywriter,
                context=[create_task]
            )

            # Assemble Crew
            crew = Crew(
                agents=[researcher, creator, copywriter],
                tasks=[research_task, create_task, copywrite_task],
                verbose=2
            )

            # Run the crew
            result = crew.kickoff()

            # Display results
            st.success("Jingle generated!")
            st.subheader("Final Polished Jingle")
            st.text(result)

            # Optional: Display intermediate results
            with st.expander("View Research Findings"):
                st.text(crew.tasks[0].output.raw_output)
            with st.expander("View Initial Creation"):
                st.text(crew.tasks[1].output.raw_output)

        except Exception as e:
            st.error(f"An error occurred while generating the jingle: {str(e)}")
