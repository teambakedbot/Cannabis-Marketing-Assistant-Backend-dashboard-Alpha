from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, List

session_store: Dict[str, List[Dict[str, str]]] = {}


# Dependency to get or create a session
def get_session(session_id: str = None):
    if session_id is None or session_id not in session_store:
        session_id = str(uuid4())
        session_store[session_id] = []
    return session_id, session_store[session_id]


from uuid import uuid4
from typing import Dict, List
from pydantic import BaseModel
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from pydantic import Field


load_dotenv()  # Load environment variables from .env
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows only the specified origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize embedding model and language model
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm = OpenAI(model="gpt-3.5-turbo")
ft_model = "ft:gpt-3.5-turbo-0125:bakedbot::9rOlft9b"
ft_llm = OpenAI(model=ft_model)


# Function to get query engine
def get_query_engine(path):
    vector_store = FaissVectorStore.from_persist_dir(path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=path
    )
    index = load_index_from_storage(
        storage_context=storage_context, embed_model=embed_model
    )
    query_engine = index.as_query_engine()
    return query_engine


# Initialize query engines
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
        description="Provides marketing plans and strategies tailored for seasonal and holiday events. ",
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
    return f"<h2>Campaign Planner Template: {template_name}</h2><p>[Template content specific to {template_name}]</p>"


def calculate_roi(investment: float, revenue: float) -> float:
    return (revenue - investment) / investment


def generate_compliance_checklist(state: str) -> str:
    return f"<h2>Compliance Checklist for {state}</h2><ul><li>Checklist item 1</li><li>Checklist item 2</li><li>Checklist item 3</li></ul>"


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


# Define the fine-tuned LLM function
def recommend_cannabis_strain(question: str) -> str:
    """
    Recommend a cannabis strain based on the given question.

    Attributes:
        question (str): A detailed question containing attributes such as type, rating, effects, and flavor.

    Returns:
        str: A detailed recommendation of a cannabis strain.
    """
    messages = [
        ChatMessage(
            role="system",
            content="You are an expert in cannabis marketing, providing detailed and personalized recommendations.",
        ),
        ChatMessage(role="user", content=question),
    ]
    resp = ft_llm.chat(messages)
    answer = resp.message.content
    return answer


# Create the FunctionTool with a detailed description
recommend_cannabis_strain_tool = FunctionTool.from_defaults(
    recommend_cannabis_strain,
    name="CannabisStrainRecommendation",
    description=(
        "This tool recommends a cannabis strain with details based on a detailed user question. "
    ),
)


# Combine all tools
tools = [
    compliance_guidelines_tool,
    marketing_strategies_tool,
    recommend_cannabis_strain_tool,
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

Additionally, you have the ability to recommend cannabis strains based on specific user-provided attributes such as type, rating, desired effects, and preferred flavors. 
The personalized strain recommendation tool helps users find the best cannabis strains that match their preferences and needs.

Always respond in HTML format.
"""


agent = OpenAIAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
)


# Define request and response models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, session: tuple = Depends(get_session)):
    session_id, chat_history = session
    try:
        user_message = request.message
        chat_history.append({"role": "user", "content": user_message})

        agent_response = agent.chat(user_message)
        if isinstance(agent_response, str):
            response_text = agent_response
        elif agent_response is not None:
            response_text = agent_response.response
        else:
            response_text = "No response available."

        chat_history.append({"role": "assistant", "content": response_text})

        # Return the full chat history as a single response
        full_conversation = "".join(
            f"<div><strong>{msg['role']}:</strong> {msg['content']}</div>"
            for msg in chat_history
        )

        return ChatResponse(response=full_conversation), {"X-Session-ID": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app with Uvicorn for performance
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
