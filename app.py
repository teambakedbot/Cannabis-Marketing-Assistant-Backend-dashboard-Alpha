import logging
import streamlit as st
from llama_index.vector_stores.faiss import FaissVectorStore

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import os

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-3.5-turbo-1106")


def get_query_engine(path):
    # load index from disk
    vector_store = FaissVectorStore.from_persist_dir(path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=path
    )

    index = load_index_from_storage(
        storage_context=storage_context, embed_model=embed_model
    )
    query_engine = index.as_query_engine()
    return query_engine


compliance_guidelines = get_query_engine("data/Compliance guidelines")
marketing_strategies = get_query_engine("data/Marketing strategies and best practices")
seasonal_marketing = get_query_engine("data/Seasonal and holiday marketing plans")
state_policies = get_query_engine("data/State-specific cannabis marketing regulations")

# Define the query engine tools with detailed descriptions
compliance_guidelines_tool = QueryEngineTool(
    query_engine=compliance_guidelines,
    metadata=ToolMetadata(
        name="Compliance_Guidelines",
        description="Provides guidelines on compliance requirements for cannabis marketing across various regions.",
    ),
)

marketing_strategies_tool = QueryEngineTool(
    query_engine=marketing_strategies,
    metadata=ToolMetadata(
        name="Marketing_Strategies",
        description="Offers strategies and best practices for effective cannabis marketing.",
    ),
)

seasonal_marketing_tool = QueryEngineTool(
    query_engine=seasonal_marketing,
    metadata=ToolMetadata(
        name="Seasonal_Marketing",
        description="Provides marketing plans and strategies tailored for seasonal and holiday events.",
    ),
)

state_policies_tool = QueryEngineTool(
    query_engine=state_policies,
    metadata=ToolMetadata(
        name="State_Policies",
        description="Details state-specific cannabis marketing regulations and policies.",
    ),
)


# Define custom tools
def generate_campaign_planner(template_name: str) -> str:
    """Generates a campaign planning template based on the user's requirements."""
    return f"Campaign Planner Template: {template_name}\n\n[Template content specific to {template_name}]"


def calculate_roi(investment: float, revenue: float) -> float:
    """Calculates the return on investment (ROI) for a given investment and revenue."""
    return (revenue - investment) / investment


def generate_compliance_checklist(state: str) -> str:
    """Generates a compliance checklist for the specified state."""
    return f"Compliance Checklist for {state}\n\n- Checklist item 1\n- Checklist item 2\n- Checklist item 3"


# Define the custom tools
campaign_planner_tool = FunctionTool.from_defaults(
    fn=generate_campaign_planner,
    name="GenerateCampaignPlanner",
    description="Generates a campaign planning template based on the user's requirements.",
)

roi_calculator_tool = FunctionTool.from_defaults(
    fn=calculate_roi,
    name="CalculateROI",
    description="Calculates the return on investment (ROI) for a given investment and revenue.",
)

compliance_checklist_tool = FunctionTool.from_defaults(
    fn=generate_compliance_checklist,
    name="GenerateComplianceChecklist",
    description="Generates a compliance checklist for the specified state.",
)

# Combine all tools
tools = [
    compliance_guidelines_tool,
    marketing_strategies_tool,
    seasonal_marketing_tool,
    state_policies_tool,
    campaign_planner_tool,
    roi_calculator_tool,
    compliance_checklist_tool,
]

system_prompt = """
You are an AI-powered chatbot specialized in assisting cannabis marketers with strategy, compliance, and campaign planning.
Your main objectives are to provide accurate, up-to-date information on cannabis marketing regulations across different US states and Canada,
offer strategic marketing advice tailored to the user's experience level, assist in campaign planning including seasonal and holiday-specific strategies,
and ensure all advice adheres to legal and ethical standards in cannabis marketing.
"""

agent = OpenAIAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
)

# Streamlit app
st.title("Cannabis Marketing Chatbot")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "session_1"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input for new messages
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Log the user prompt
    logger.debug(f"User prompt: {prompt}")

    # Send the message to the API and get the response
    logger.debug("Sending prompt to agent...")
    response = agent.chat(prompt)
    if response is None:
        response = "No response available."
        logger.error("Agent failed to provide a response.")
    else:
        logger.debug(f"Agent response: {response}")

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
