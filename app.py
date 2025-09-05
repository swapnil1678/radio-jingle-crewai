# app.py
import streamlit as st
import sys
from tenacity import retry, stop_after_attempt, wait_exponential

# Workaround for sqlite3 version issue
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from crewai import Agent, Task, Crew
from langchain_huggingface import HuggingFaceEndpoint
import os

# Set up the page
st.title("Radio Jingle Generator using CrewAI and SLM")
st.markdown("""
This app uses CrewAI with a Small Language Model (Mistral 7B) to generate short radio jingles. It includes:
- **Researcher AI**: Researches the theme or product.
- **Creator AI**: Creates the initial jingle lyrics and structure.
- **Copywriter AI**: Polishes and refines the content for radio.

Enter a theme (e.g., "coffee shop promotion") and generate!
""")

# Hugging Face API Key handling
hf_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_key:
    st.error("Hugging Face API key not found. Please set it in Streamlit secrets or environment variables.")
    st.stop()

# Check pip version
import pip
pip_version = pip.__version__
if pip_version < "25.2":
    st.warning(f"Using pip {pip_version}. Consider updating to 25.2 or later for better dependency resolution.")

# LLM setup with Mistral 7B
try:
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mixtral-8x7b-instruct-v0.1",
        huggingfacehub_api_token=hf_api_key,
        max_new_tokens=500,  # Limit output length
        temperature=0.7,
        top_p=0.9
    )
except Exception as e:
    st.error(f"Failed to initialize SLM: {str(e)}")
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
                description=f"Research '{theme}'. List 5 key points for an engaging jingle.",
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
                verbose=True
            )

            # Run the crew with retry logic
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
            def run_crew():
                return crew.kickoff()

            result = run_crew()

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
