import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="AI 팀 워크플로우", layout="wide")

# Initialize session state
if 'crew' not in st.session_state:
    st.session_state.crew = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Function to create agents
def create_agents(topic):
    researcher = Agent(
        role='Senior Researcher',
        goal=f'Discover groundbreaking technologies in {topic}',
        backstory="You're at the forefront of innovation, eager to explore and share knowledge that could change the world.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    )

    analyst = Agent(
        role='Data Analyst',
        goal=f'Analyze research findings and identify key trends in {topic}',
        backstory="You excel at interpreting complex data and presenting actionable insights.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    )

    writer = Agent(
        role='Technical Writer',
        goal=f'Craft an engaging technological narrative about {topic}',
        backstory="You're skilled at simplifying complex subjects and creating captivating narratives.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)
    )

    editor = Agent(
        role='Chief Editor',
        goal='Ensure the final report is polished, coherent, and of high quality',
        backstory="You have a sharp eye for detail and are committed to producing professional content.",
        verbose=True,
        allow_delegation=True,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    )

    return [researcher, analyst, writer, editor]

# Function to create tasks
def create_tasks(agents, topic):
    return [
        Task(
            description=f"Conduct comprehensive research on {topic}. Provide detailed findings and cite sources.",
            agent=agents[0],
            expected_output="Detailed research report with cited sources"
        ),
        Task(
            description=f"Analyze the research findings on {topic}. Identify key trends, patterns, and insights.",
            agent=agents[1],
            expected_output="Analysis report highlighting key trends and insights"
        ),
        Task(
            description=f"Write a compelling report on {topic} based on the research and analysis. Ensure it's engaging and informative.",
            agent=agents[2],
            expected_output="Engaging and informative report on the topic"
        ),
        Task(
            description="Review and refine the final report. Ensure clarity, coherence, and overall quality.",
            agent=agents[3],
            expected_output="Polished and high-quality final report"
        )
    ]

# Function to initialize crew
def initialize_crew(topic):
    agents = create_agents(topic)
    tasks = create_tasks(agents, topic)
    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=2,
        process=Process.sequential
    )

# Function to display conversation
def display_conversation():
    for entry in st.session_state.conversation:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

# Streamlit UI
st.title("AI 팀 워크플로우")

# Sidebar for user input and API key
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.header("작업 입력")
    user_input = st.text_area("연구 주제를 입력하세요:", height=100, 
                              placeholder="예: '한국의 현대 페미니즘 운동의 현황'")
    start_button = st.button('작업 시작', key='start')

# Main content area
if start_button and user_input:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해주세요.")
    else:
        # Initialize crew
        st.session_state.crew = initialize_crew(user_input)
        st.session_state.conversation = []

        with st.spinner('AI 팀이 작업을 수행하고 있습니다... 🤔💬'):
            try:
                result = st.session_state.crew.kickoff()
                st.session_state.result = result
                st.success("작업이 완료되었습니다! 최종 보고서는 다음과 같습니다:")
                st.markdown(st.session_state.result)
            except Exception as e:
                st.error(f"작업 수행 중 오류가 발생했습니다: {str(e)}")

# Option for follow-up questions
if st.session_state.result:
    user_question = st.text_input("보고서에 대해 궁금한 점이 있으신가요?")
    if user_question:
        st.session_state.conversation.append({"role": "user", "content": user_question})
        display_conversation()
        
        with st.spinner('답변을 생성하고 있습니다... 🤔💬'):
            try:
                answer_agent = Agent(
                    role='QA Specialist',
                    goal='Answer questions about the report accurately and concisely',
                    backstory="You're an expert on the report and can answer questions about it in detail.",
                    verbose=True,
                    allow_delegation=False,
                    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
                )
                
                answer_task = Task(
                    description=f"Answer the following question about the report in Korean: {user_question}\n\nUse the following report as context:\n{st.session_state.result}",
                    agent=answer_agent,
                    expected_output="Concise and accurate answer to the user's question"
                )
                
                answer_crew = Crew(
                    agents=[answer_agent],
                    tasks=[answer_task],
                    verbose=2,
                    process=Process.sequential
                )
                
                response = answer_crew.kickoff()
                
                st.session_state.conversation.append({"role": "assistant", "content": response})
                display_conversation()
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")