from typing import List, Dict, Any, Optional
from functools import lru_cache

# Core components
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.tools import Tool

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Main langchain components
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chains import RetrievalQA

# Vector store and embeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Project-specific imports
from ..config.config import settings, logger
from ..utils.redis_config import FirestoreEncoder
from ..models.schemas import Product

# Third-party imports
from openai import OpenAI as OpenAI2
from firebase_admin import storage
from redis.asyncio import Redis
import os
import requests
import tempfile
import time
import asyncio
import uuid
import json
from typing import Dict, Any, Optional, List, TypedDict

# Initialize core components
llm = ChatOpenAI(
    model_name=settings.OPENAI_MODEL_NAME, temperature=0.1, max_tokens=4096
)
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone
pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
index_name = "knowledge-index"

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric="cosine")

# Get the index
index = pc.Index(index_name)

# Create the vector store
embeddings = OpenAIEmbeddings()
vector_store = Pinecone.from_existing_index(index_name, embeddings)

# Initialize Redis client
redis_client = Redis.from_url(
    settings.REDISCLOUD_URL, encoding="utf-8", decode_responses=True
)


# Helper functions
def get_retriever(namespace: str):
    """Create a retriever for a specific namespace in Pinecone."""
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace
    )
    return vectorstore.as_retriever()


def add_disclaimer(response, disclaimer_type="general"):
    """Add appropriate disclaimers to responses."""
    disclaimers = {
        "legal": "\n\n*Please note: This information is provided for general informational purposes only and should not be considered legal advice.*",
        "medical": "\n\n*Please note: This information is provided for general informational purposes only and should not be considered medical advice. Consult a healthcare professional for medical concerns.*",
        "general": "\n\n*Please note: This information is provided for general informational purposes only.*",
    }
    return response + disclaimers.get(disclaimer_type, disclaimers["general"])


@lru_cache(maxsize=100)
def get_retailer_info(query: str) -> str:
    """
    Retrieve retailer information from the Pinecone vector database based on the user's query.
    """
    logger.info(f"Fetching retailer information for query: {query}")

    try:
        query_embedding = embed_model.embed_query(query)

        index = pc.Index("retailer-index")
        response = index.query(
            vector=query_embedding, top_k=1, include_values=False, include_metadata=True
        )

        if not response["matches"]:
            return json.dumps(
                {"error": f"No retailer found matching: {query}"}, cls=FirestoreEncoder
            )

        retailer_data = response["matches"][0]["metadata"]
        response_data = {
            "retailer_id": retailer_data.get("retailer_id"),
            "name": retailer_data.get("retailer_name"),
            "address": retailer_data.get("address"),
            "city": retailer_data.get("city"),
            "state": retailer_data.get("state"),
            "zip_code": retailer_data.get("zip_code"),
            "phone": retailer_data.get("phone"),
            "email": retailer_data.get("email"),
            "website": retailer_data.get("website"),
            "license_number": retailer_data.get("license_number"),
            "serves_recreational_users": retailer_data.get("serves_recreational_users"),
            "serves_medical_users": retailer_data.get("serves_medical_users"),
            "latitude": retailer_data.get("latitude"),
            "longitude": retailer_data.get("longitude"),
        }
        logger.debug(f"Retailer information found: {response_data}")
        return json.dumps(response_data, indent=2, cls=FirestoreEncoder)

    except Exception as e:
        logger.error(f"Error fetching retailer information: {e}")
        return json.dumps(
            {
                "error": "An unexpected error occurred while fetching retailer information. Please try again later.",
                "query": query,
            },
            cls=FirestoreEncoder,
        )


@lru_cache(maxsize=100)
def get_compliance_guidelines(query: str) -> str:
    response = compliance_guidelines_chain.run(query)
    return add_disclaimer(response, "legal")


@lru_cache(maxsize=100)
def provide_medical_information(query: str) -> str:
    response = medical_information_chain.run(query)
    return add_disclaimer(response, "medical")


@lru_cache(maxsize=100)
def generate_campaign_planner(template_name: str) -> str:
    logger.debug(f"Generating campaign planner for template: {template_name}")
    return f"<h2>Campaign Planner Template: {template_name}</h2><p>[Template content specific to {template_name}]</p>"


def calculate_roi(investment: float, revenue: float) -> float:
    logger.debug(f"Calculating ROI for investment: {investment}, revenue: {revenue}")
    roi = (revenue - investment) / investment
    logger.debug(f"Calculated ROI: {roi}")
    return roi


@lru_cache(maxsize=100)
def generate_compliance_checklist(state: str) -> str:
    checklist = f"""<h2>Compliance Checklist for {state}</h2>
    <ul>
    <li>Checklist item 1</li>
    <li>Checklist item 2</li>
    <li>Checklist item 3</li>
    </ul>"""
    return add_disclaimer(checklist, "legal")


@lru_cache(maxsize=100)
def recommend_cannabis_strain(question: str) -> list:
    """Recommend a cannabis strain based on the given question."""
    messages = [
        SystemMessage(
            content="You are an expert in cannabis marketing, providing detailed and personalized recommendations."
        ),
        HumanMessage(content=question),
    ]
    resp = llm(messages)
    answer_parts = resp.content.split("\n\n", 1)
    return answer_parts if len(answer_parts) > 1 else [resp.content, ""]


def get_products_from_db(query: str) -> List[Product]:
    try:
        query_embedding = embed_model.embed_query(query)
        index = pc.Index("product-index")
        response = index.query(
            vector=query_embedding,
            top_k=10,
            include_values=False,
            include_metadata=True,
        )
        recommended_products = []
        for product in response["matches"]:
            try:
                metadata = product.metadata
                product_data = {
                    "id": product.id,
                    "cann_sku_id": metadata.get("cann_sku_id", ""),
                    "product_name": metadata.get("product_name", ""),
                    "brand_name": metadata.get("brand_name", ""),
                    "category": metadata.get("category", ""),
                    "raw_product_category": metadata.get("raw_product_category", ""),
                    "image_url": metadata.get("image_url", ""),
                    "latest_price": float(metadata.get("latest_price", 0)),
                    "display_weight": metadata.get("display_weight", ""),
                    "percentage_thc": float(metadata.get("percentage_thc", 0) or 0),
                    "percentage_cbd": float(metadata.get("percentage_cbd", 0) or 0),
                    "mg_thc": float(metadata.get("mg_thc", 0) or 0),
                    "mg_cbd": float(metadata.get("mg_cbd", 0) or 0),
                    "subcategory": metadata.get("subcategory", ""),
                    "raw_subcategory": metadata.get("raw_subcategory", ""),
                    "product_tags": metadata.get("product_tags", []),
                    "medical": bool(metadata.get("medical", False)),
                    "recreational": bool(metadata.get("recreational", False)),
                    "menu_provider": metadata.get("menu_provider", ""),
                    "retailer_id": metadata.get("retailer_id", ""),
                    "meta_sku": metadata.get("meta_sku", ""),
                    "raw_product_name": metadata.get("raw_product_name", ""),
                }

                if isinstance(product_data["product_tags"], str):
                    try:
                        product_data["product_tags"] = eval(
                            product_data["product_tags"]
                        )
                    except:
                        product_data["product_tags"] = []

                recommended_product = Product(**product_data)
                recommended_products.append(recommended_product)
            except Exception as e:
                logger.error(f"Error processing individual product: {e}")
                logger.error(f"Problematic product data: {metadata}")

        return recommended_products

    except Exception as e:
        logger.error(f"Error querying products: {e}")
        return []


def get_products_with_error_handling(query: str) -> tuple:
    try:
        print(f"Querying products for: {query}")
        products = get_products_from_db(query)
        if not products:
            return "No products found", json.dumps(
                {"error": "No products found", "products": []}
            )
        product_json = json.dumps(
            {"products": [product.dict() for product in products]}
        )
        return f"Found {len(products)} products matching your query.", product_json
    except Exception as e:
        logger.error(f"Error in product recommendation: {e}")
        return f"An error occurred: {str(e)}", json.dumps(
            {"error": str(e), "products": []}
        )


def store_image_in_firebase(
    image_url: str, prompt: str, user_id: Optional[str] = None
) -> str:
    """Downloads an image from a URL and stores it in Firebase Storage."""
    if not user_id:
        logger.warning("Unauthorized attempt to generate image: no user_id provided")
        return ""

    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()

        # Create temp file
        suffix = ".png"  # or determine from content-type
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        # Upload to Firebase Storage
        bucket = storage.bucket(name=settings.FIREBASE_STORAGE_BUCKET)
        timestamp = int(time.time())
        filename = "".join(
            x if x.isalnum() or x in ("_", "-") else "_" for x in prompt
        )[:50]
        blob_path = f"temp_images/{user_id}/{filename}_{timestamp}{suffix}"
        blob = bucket.blob(blob_path)

        blob.upload_from_filename(temp_path)
        blob.make_public()

        # Clean up temp file
        os.unlink(temp_path)

        return blob.public_url
    except Exception as e:
        logger.error(f"Error storing image in Firebase: {str(e)}")
        return image_url  # Fallback to original URL if storage fails


def generate_image_with_dalle(prompt: str, config: Dict[str, Any] = None) -> str:
    """Generate an image using DALL-E based on the given prompt."""
    user_id = config.get("user_id") if config else None
    if not user_id:
        return "Image generation is only available for authenticated users."

    cache_key = f"dalle_image:{user_id}:{prompt}"
    cached_image = redis_client.get(cache_key)
    if cached_image:
        return cached_image

    try:
        logger.info(
            f"Generating image with DALL-E for user {user_id}, prompt: {prompt}"
        )

        client = OpenAI2(api_key=settings.OPENAI_API_KEY)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url",
        )

        image_url = response.data[0].url
        logger.info("Image generated successfully")

        # Store in Firebase and get permanent URL
        permanent_url = store_image_in_firebase(image_url, prompt, user_id)
        if not permanent_url:
            return "Failed to store image. Please try again."

        redis_client.set(cache_key, permanent_url, ex=3600)  # Cache for 1 hour
        return permanent_url
    except Exception as e:
        logger.error(f"Error generating image with DALL-E: {str(e)}")
        return f"Error generating image: {str(e)}"


def generate_image_with_ideogram(prompt: str, config: Dict[str, Any] = None) -> str:
    """Generate an image using Ideogram based on the given prompt."""
    user_id = config.get("user_id") if config else None
    if not user_id:
        return "Image generation is only available for authenticated users."

    cache_key = f"ideogram_image:{user_id}:{prompt}"
    cached_image = redis_client.get(cache_key)
    if cached_image:
        return cached_image

    try:
        logger.info(
            f"Generating image with Ideogram for user {user_id}, prompt: {prompt}"
        )

        url = "https://api.ideogram.ai/generate"
        payload = json.dumps(
            {
                "image_request": {
                    "prompt": prompt,
                    "style": "photo",
                    "aspectRatio": "1:1",
                    "magic_prompt_option": "AUTO",
                }
            },
            cls=FirestoreEncoder,
        )
        headers = {
            "Content-Type": "application/json",
            "Api-Key": f"{settings.IDEOGRAM_API_KEY}",
        }

        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()

        data = response.json()
        image_url = data["data"][0]["url"]
        logger.info("Image generated successfully with Ideogram")

        # Store in Firebase and get permanent URL
        permanent_url = store_image_in_firebase(image_url, prompt, user_id)
        if not permanent_url:
            return "Failed to store image. Please try again."

        redis_client.set(cache_key, permanent_url, ex=3600)  # Cache for 1 hour
        return permanent_url
    except Exception as e:
        logger.error(f"Error generating image with Ideogram: {str(e)}")
        return f"Error generating image: {str(e)}"


# Initialize retrievers
compliance_guidelines_retriever = get_retriever("Compliance guidelines")
marketing_strategies_retriever = get_retriever(
    "Marketing strategies and best practices"
)
seasonal_marketing_retriever = get_retriever("Seasonal and holiday marketing plans")
state_policies_retriever = get_retriever(
    "State-specific cannabis marketing regulations"
)
general_knowledge_retriever = get_retriever("General Cannabis Knowledge")
usage_instructions_retriever = get_retriever("Cannabis Usage Instructions")

# Create RetrievalQA chains
compliance_guidelines_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=compliance_guidelines_retriever
)

marketing_strategies_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=marketing_strategies_retriever
)

seasonal_marketing_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=seasonal_marketing_retriever
)

state_policies_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=state_policies_retriever
)

general_knowledge_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=general_knowledge_retriever
)

usage_instructions_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=usage_instructions_retriever
)

# Create tools
retailer_info_tool = Tool(
    name="RetailerInformation",
    func=get_retailer_info,
    description=(
        "Use this tool to find detailed information about cannabis retailers. "
        "Provides location details, contact information, operating hours, and services offered. "
        "Can search by name, location, or other criteria."
    ),
    return_direct=False,
    verbose=True,
    handle_tool_error=True,
)


# Create specialized tools with better error handling and response formatting
def create_tool(name: str, chain: Any, description: str) -> Tool:
    """Create a tool with error handling and response formatting."""

    def tool_func(query: str) -> str:
        try:
            response = chain.run(query)
            # Add disclaimer based on tool type
            if "compliance" in name.lower() or "policies" in name.lower():
                response = add_disclaimer(response, "legal")
            elif "medical" in name.lower():
                response = add_disclaimer(response, "medical")
            return response
        except Exception as e:
            logger.error(f"Error in {name} tool: {e}")
            return f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question or contact support if the issue persists."

    return Tool(
        name=name,
        func=tool_func,
        description=description,
        return_direct=False,
        verbose=True,
        handle_tool_error=True,
    )


# Create the tools with better descriptions and error handling
compliance_guidelines_tool = create_tool(
    "Compliance_Guidelines",
    compliance_guidelines_chain,
    "Use this tool for questions about cannabis marketing compliance requirements, regulations, and guidelines. Provides detailed compliance information with citations.",
)

marketing_strategies_tool = create_tool(
    "Marketing_Strategies",
    marketing_strategies_chain,
    "Use this tool for questions about effective cannabis marketing strategies, best practices, and campaign ideas. Provides actionable marketing advice that is compliant with regulations.",
)

seasonal_marketing_tool = create_tool(
    "Seasonal_Marketing",
    seasonal_marketing_chain,
    "Use this tool for seasonal and holiday-specific cannabis marketing ideas and campaign planning. Provides creative, timely, and compliant marketing strategies.",
)

state_policies_tool = create_tool(
    "State_Policies",
    state_policies_chain,
    "Use this tool for state-specific cannabis marketing regulations, requirements, and policy information. Provides detailed state-by-state compliance guidance.",
)

general_knowledge_tool = create_tool(
    "General_Cannabis_Knowledge",
    general_knowledge_chain,
    "Provides general information about cannabis, including effects, usage, and terminology.",
)

usage_instructions_tool = create_tool(
    "Usage_Instructions",
    usage_instructions_chain,
    "Provides step-by-step instructions on how to use different cannabis products and consumption methods.",
)

campaign_planner_tool = Tool(
    name="GenerateCampaignPlanner",
    func=generate_campaign_planner,
    description="Generates a campaign planning template based on the user's requirements.",
)

roi_calculator_tool = Tool(
    name="CalculateROI",
    func=calculate_roi,
    description="Calculates the return on investment (ROI) for a given investment and revenue.",
)

compliance_checklist_tool = Tool(
    name="GenerateComplianceChecklist",
    func=generate_compliance_checklist,
    description="Generates a compliance checklist for the specified state.",
)

recommend_cannabis_strain_tool = Tool(
    name="CannabisStrainRecommendation",
    func=recommend_cannabis_strain,
    description="This tool recommends a cannabis strain with details based on a detailed user question.",
)

product_recommendation_tool = Tool(
    name="ProductRecommendation",
    func=get_products_with_error_handling,
    description="Use this tool to retrieve cannabis products based on user queries. It returns a tuple containing a message and a JSON string with a list of recommended products or an error message.",
    return_direct=True,
)


# Define the state schema for the chat graph
class MessagesState(TypedDict):
    """State definition for the chat graph."""

    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]


class ConfigurableAgent:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.checkpointer = MemorySaver()
        self.tools = self._create_tools()
        self.workflow = self._create_workflow()
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def _create_tools(self):
        """Create and return the list of available tools."""

        # Create wrapped versions of the image generation functions that include config
        def generate_dalle_with_config(prompt: str) -> str:
            return generate_image_with_dalle(prompt, self.config)

        def generate_ideogram_with_config(prompt: str) -> str:
            return generate_image_with_ideogram(prompt, self.config)

        tools = [
            compliance_guidelines_tool,
            marketing_strategies_tool,
            seasonal_marketing_tool,
            state_policies_tool,
            general_knowledge_tool,
            usage_instructions_tool,
            retailer_info_tool,
            campaign_planner_tool,
            roi_calculator_tool,
            compliance_checklist_tool,
            recommend_cannabis_strain_tool,
            product_recommendation_tool,
            Tool(
                name="GenerateImageWithDALLE",
                func=generate_dalle_with_config,
                description=(
                    "Generates an image using DALL-E based on a text description. "
                    "Use this tool when a user requests an image to be created or visualized."
                ),
            ),
            Tool(
                name="GenerateImageWithIdeogram",
                func=generate_ideogram_with_config,
                description=(
                    "Generates a photorealistic image using Ideogram based on a text description. "
                    "Use this tool when a user requests a realistic image to be created or visualized."
                ),
            ),
        ]
        return tools

    def _create_workflow(self):
        """Create the workflow graph with persistence."""
        workflow = StateGraph(state_schema=MessagesState)

        # Define the function that calls the model
        def call_model(state: MessagesState):
            messages = state["messages"]
            system_prompt = (
                "You are a helpful assistant specializing in cannabis marketing. "
                "Answer all questions to the best of your ability using the available tools. "
                "Keep responses concise unless asked for more detail. "
                "Always prioritize legal compliance and responsible use."
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    *[(msg["role"], msg["content"]) for msg in messages],
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Create agent with tools
            agent = create_openai_functions_agent(
                llm=llm, tools=self.tools, prompt=prompt
            )

            # Create executor
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                handle_parsing_errors=True,
            )

            # Execute agent
            response = agent_executor.invoke(
                {
                    "input": messages[-1]["content"] if messages else "",
                    "chat_history": messages[:-1] if len(messages) > 1 else [],
                }
            )

            return {
                "messages": messages
                + [{"role": "assistant", "content": response["output"]}]
            }

        # Add nodes and edges
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        workflow.add_edge("model", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def ainvoke(
        self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ):
        try:
            if config:
                self.config.update(config.get("configurable", {}))

            # Ensure we have a thread_id
            if "thread_id" not in self.config.get("configurable", {}):
                self.config["configurable"] = {
                    **(self.config.get("configurable", {})),
                    "thread_id": str(uuid.uuid4()),
                }

            # Format input for the workflow
            messages = inputs.get("messages", [])
            workflow_input = {
                "messages": messages,
                "metadata": {
                    "user_id": self.config.get("configurable", {}).get("user_id"),
                    "session_id": self.config.get("configurable", {}).get("session_id"),
                },
            }

            # Execute workflow
            result = await self.workflow.ainvoke(workflow_input, config=self.config)

            return result

        except Exception as e:
            logger.error(f"Error in ainvoke: {str(e)}")
            return {
                "messages": messages
                + [
                    {
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error. Please try again.",
                    }
                ],
                "error": str(e),
            }

    def get_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the current state of the conversation."""
        return self.workflow.get_state(config)

    def get_state_history(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the history of states for this conversation."""
        return list(self.workflow.get_state_history(config))

    async def _handle_error(self, error: Exception, attempt: int) -> Dict[str, Any]:
        """Handle errors during workflow execution."""
        logger.error(f"Workflow error on attempt {attempt}: {str(error)}")

        if attempt < self.max_retries:
            await asyncio.sleep(self.retry_delay * attempt)
            return {"error": "temporary", "should_retry": True}

        return {
            "error": str(error),
            "should_retry": False,
            "messages": [
                {
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error. Please try again.",
                }
            ],
        }


# Create a single instance of the configurable agent
configurable_agent = ConfigurableAgent()
