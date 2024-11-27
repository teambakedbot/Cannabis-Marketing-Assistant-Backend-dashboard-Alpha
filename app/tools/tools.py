from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain.schema import SystemMessage, HumanMessage
from ..config.config import settings
from openai import OpenAI as OpenAI2
import base64
import os
import requests
from ..utils.redis_config import FirestoreEncoder
from ..config.config import logger
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
import json
from functools import lru_cache
from redis import Redis
from langchain_core.output_parsers import PydanticOutputParser
from ..models.schemas import Product
from typing import List
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Dict, Any

from langchain.chains import ConversationChain
from langchain.memory.chat_memory import BaseChatMemory, BaseMemory


# Initialize Redis client
# redis_client = Redis(host="localhost", port=6379, db=0)
redis_client = Redis.from_url(
    settings.REDISCLOUD_URL, encoding="utf-8", decode_responses=True
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

pinecone_api_key = settings.PINECONE_API_KEY
index_name = "knowledge-index"

embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone
pc = PineconeClient(api_key=settings.PINECONE_API_KEY)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric="cosine")

# Get the index
index = pc.Index(index_name)

# Create the vector store
embeddings = OpenAIEmbeddings()
vector_store = Pinecone.from_existing_index(index_name, embeddings)


@lru_cache(maxsize=100)
def get_compliance_guidelines(query: str) -> str:
    response = compliance_guidelines.query(query)
    return add_disclaimer(response.response, "legal")


@lru_cache(maxsize=100)
def provide_medical_information(query):
    response = medical_information.query(query)
    return add_disclaimer(response, "medical")


def add_disclaimer(response, disclaimer_type="general"):
    disclaimers = {
        "legal": "\n\n*Please note: This information is provided for general informational purposes only and should not be considered legal advice.*",
        "medical": "\n\n*Please note: This information is provided for general informational purposes only and should not be considered medical advice. Consult a healthcare professional for medical concerns.*",
        "general": "\n\n*Please note: This information is provided for general informational purposes only.*",
    }
    return response + disclaimers.get(disclaimer_type, disclaimers["general"])


def get_retriever(namespace: str):
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace
    )
    return vectorstore.as_retriever()


# Initialize query engines with Pinecone
compliance_guidelines = get_retriever("Compliance guidelines")
marketing_strategies = get_retriever("Marketing strategies and best practices")
seasonal_marketing = get_retriever("Seasonal and holiday marketing plans")
state_policies = get_retriever("State-specific cannabis marketing regulations")

compliance_guidelines_retriever = get_retriever("Compliance guidelines")
compliance_guidelines_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=compliance_guidelines_retriever
)
compliance_guidelines_tool = Tool(
    name="Compliance_Guidelines",
    func=compliance_guidelines_chain.run,
    description="Provides guidelines on compliance requirements for cannabis marketing across various regions.",
)

marketing_strategies_retriever = get_retriever(
    "Marketing strategies and best practices"
)
marketing_strategies_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=marketing_strategies_retriever
)
marketing_strategies_tool = Tool(
    name="Marketing_Strategies",
    func=marketing_strategies_chain.run,
    description="Offers strategies and best practices for effective cannabis marketing.",
)

seasonal_marketing_retriever = get_retriever("Seasonal and holiday marketing plans")
seasonal_marketing_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=seasonal_marketing_retriever
)
seasonal_marketing_tool = Tool(
    name="Seasonal_Marketing",
    func=seasonal_marketing_chain.run,
    description="Provides marketing plans and strategies tailored for seasonal and holiday events.",
)


@lru_cache(maxsize=100)
def get_state_policies(query: str) -> str:
    response = state_policies.query(query)
    return add_disclaimer(response.response, "legal")


state_policies_retriever = get_retriever(
    "State-specific cannabis marketing regulations"
)
state_policies_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=state_policies_retriever
)
state_policies_tool = Tool(
    name="State_Policies",
    func=state_policies_chain.run,
    description="Details state-specific cannabis marketing regulations and policies.",
)


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


@lru_cache(maxsize=100)
def recommend_cannabis_strain(question: str) -> list:
    """
    Recommend a cannabis strain based on the given question.
    """
    messages = [
        SystemMessage(
            content="You are an expert in cannabis marketing, providing detailed and personalized recommendations."
        ),
        HumanMessage(content=question),
    ]
    resp = llm(messages)
    answer_parts = resp.content.split("\n\n", 1)
    return answer_parts if len(answer_parts) > 1 else [resp.content, ""]


recommend_cannabis_strain_tool = Tool(
    name="CannabisStrainRecommendation",
    func=recommend_cannabis_strain,
    description="This tool recommends a cannabis strain with details based on a detailed user question.",
)


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


product_recommendation_tool = Tool(
    name="ProductRecommendation",
    func=get_products_with_error_handling,
    description="Use this tool to retrieve cannabis products based on user queries. It returns a tuple containing a message and a JSON string with a list of recommended products or an error message.",
    return_direct=True,
)


@lru_cache(maxsize=100)
def get_cached_retailer_info(query: str) -> str:
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


retailer_info_tool = Tool(
    name="RetailerInformation",
    func=get_cached_retailer_info,
    description="Retrieves cached detailed information about a cannabis retailer based on the user's query.",
)

general_knowledge = get_retriever("General Cannabis Knowledge")

general_knowledge_retriever = get_retriever("General Cannabis Knowledge")

general_knowledge_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=general_knowledge_retriever
)

general_knowledge_tool = Tool(
    name="General_Cannabis_Knowledge",
    func=general_knowledge_chain.run,
    description="Provides general information about cannabis, including effects, usage, and terminology.",
)

usage_instructions = get_retriever("Cannabis Usage Instructions")

usage_instructions_retriever = get_retriever("Cannabis Usage Instructions")

usage_instructions_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=usage_instructions_retriever
)

usage_instructions_tool = Tool(
    name="Usage_Instructions",
    func=usage_instructions_chain.run,
    description="Provides step-by-step instructions on how to use different cannabis products and consumption methods.",
)


@lru_cache(maxsize=100)
def medical_information(query: str) -> str:
    """
    Provide general information about medical cannabis use.
    """
    response = f"Here is some general information about medical cannabis use related to your query: '{query}'"
    return add_disclaimer(response, "medical")


medical_information_tool = Tool(
    name="MedicalInformation",
    func=medical_information,
    description="Provides general information about medical cannabis. Does not offer medical advice.",
)


def generate_image_with_dalle(prompt: str) -> str:
    """
    Generate an image using DALL-E based on the given prompt.
    """
    cache_key = f"dalle_image:{prompt}"
    cached_image = redis_client.get(cache_key)
    if cached_image:
        return cached_image

    try:
        logger.info(f"Generating image with DALL-E for prompt: {prompt}")

        client = OpenAI2(api_key=settings.OPENAI_API_KEY)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url",
        )

        image_data = response.data[0].url
        logger.info("Image generated successfully")

        redis_client.set(cache_key, image_data, ex=3600)  # Cache for 1 hour
        return image_data
    except Exception as e:
        logger.error(f"Error generating image with DALL-E: {str(e)}")
        return ""


image_generation_tool = Tool(
    name="GenerateImageWithDALLE",
    func=generate_image_with_dalle,
    description=(
        "Generates an image using DALL-E based on a text description. "
        "Use this tool when a user requests an image to be created or visualized."
    ),
)


def generate_image_with_ideogram(prompt: str) -> str:
    """
    Generate an image using Ideogram based on the given prompt.
    """
    cache_key = f"ideogram_image:{prompt}"
    cached_image = redis_client.get(cache_key)
    if cached_image:
        return cached_image

    try:
        logger.info(f"Generating image with Ideogram for prompt: {prompt}")

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
        response.raise_for_status()  # Raises an HTTPError for bad responses

        data = response.json()
        image_url = data["data"][0]["url"]
        logger.info("Image generated successfully with Ideogram")

        redis_client.set(cache_key, image_url, ex=3600)  # Cache for 1 hour
        return image_url
    except Exception as e:
        logger.error(f"Error generating image with Ideogram: {str(e)}")
        return ""


ideogram_image_generation_tool = Tool(
    name="GenerateImageWithIdeogram",
    func=generate_image_with_ideogram,
    description=(
        "Generates a photorealistic image using Ideogram based on a text description. "
        "Use this tool when a user requests a realistic image to be created or visualized."
    ),
)

# Define few-shot examples
few_shot_examples = [
    {
        "input": "Show me new deals",
        "output": "Here are the latest deals:\n1. Product A - $10\n2. Product B - $15\n3. Product C - $20",
    },
    {
        "input": "I'm looking for the cheapest products",
        "output": "Sure! Here are some of our most affordable products:\n1. Product D - $5\n2. Product E - $8\n3. Product F - $12",
    },
    {
        "input": "Find me products on sale",
        "output": "Absolutely! Check out these products currently on sale:\n1. Product G - $9 (20% off)\n2. Product H - $14 (15% off)\n3. Product I - $7 (25% off)",
    }
]

# Create formatted few-shot examples
formatted_few_shot = "\n".join(
    [
        f"User: {example['input']}\nAssistant: {example['output']}"
        for example in few_shot_examples
    ]
)

system_message = SystemMessagePromptTemplate.from_template(
    f"""
You are an AI assistant specializing in cannabis marketing and compliance.
Your role is to provide accurate, helpful, and compliant information about cannabis products, 
marketing strategies, and regulations. Always prioritize legal compliance and responsible use. 
Use the tools provided to access specific information and generate responses.

### Few-Shot Examples:
{formatted_few_shot}

### Instructions:
When a user asks for product recommendations, sales, or deals, respond with a structured list of products based on the request.
Always include disclaimers where necessary.
"""
)

human_message = HumanMessagePromptTemplate.from_template("{input}")

agent_scratchpad = MessagesPlaceholder(variable_name="agent_scratchpad")

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message, human_message, agent_scratchpad]
)

tools = [
    compliance_guidelines_tool,
    marketing_strategies_tool,
    seasonal_marketing_tool,
    state_policies_tool,
    campaign_planner_tool,
    roi_calculator_tool,
    compliance_checklist_tool,
    recommend_cannabis_strain_tool,
    product_recommendation_tool,
    retailer_info_tool,
    general_knowledge_tool,
    usage_instructions_tool,
    medical_information_tool,
    image_generation_tool,
    ideogram_image_generation_tool,
]


class StatusCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("\n🤔 Agent is thinking...")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("💡 Agent has formulated a response.")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        print(f"🔧 Agent is using tool: {serialized['name']}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print("✅ Tool execution completed.")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"🏃‍♂️ Agent has decided to take action: {action.tool}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        print("🎉 Agent has finished its task.")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        print("🔗 Starting a new chain of thought...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        print("🔗 Chain of thought completed.")


# Create a callback manager with our custom StatusCallbackHandler
callback_manager = CallbackManager([StatusCallbackHandler()])

# Create the agent with the callback manager
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=chat_prompt)

# Create the agent executor with the callback manager
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools=tools, checkpointer=memory, debug=True)
