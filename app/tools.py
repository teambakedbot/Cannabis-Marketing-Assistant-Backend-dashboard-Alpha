from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain.schema import SystemMessage, HumanMessage
from .config import settings
from openai import OpenAI as OpenAI2
import base64
import os
import requests
from .redis_config import FirestoreEncoder
from .config import logger
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
from .schemas import Product
from typing import List

# Initialize Redis client
# redis_client = Redis(host="localhost", port=6379, db=0)
redis_url = os.getenv("REDISCLOUD_URL", "redis://localhost:6379")
redis_client = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)

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


def format_product_info(product):
    """
    Format product information, handling cases where data might be missing
    """
    name = product.get("name", "Product name not available")
    brand = product.get("brand", "Brand not specified")
    category = product.get("category", "Category not specified")
    thc = product.get("thc")
    cbd = product.get("cbd")
    price = product.get("price")
    strain = product.get("strain", "Strain not specified")
    image = product.get(
        "image_url",
        "https://images.weedmaps.com/categories/000/000/003/placeholder/1613661821-1613605716-concentrates_image_missing.jpg",
    )

    thc_info = f"THC: {thc}%" if thc is not None else ""
    cbd_info = f"CBD: {cbd}%" if cbd is not None else ""
    price_info = f"Price: ${price}" if price is not None else ""

    return (
        f"<div class='product-info onclick='window.location.href=\"/products/{product.get('retailer_id')}\"'>\n"
        f"**{name}** by {brand} - Category: {category}\n"
        f"{thc_info}, {cbd_info}\n"
        f"{price_info}\n"
        f"Strain: {strain}\n"
        f"![Product Image]({image}){{.product-image}}\n"
        f"</div>"
    )


def get_products_from_db(
    query: str, max_price: float = None, product_type: str = None
) -> str:
    """
    Retrieve product information from the Pinecone vector database based on the user's query.
    """
    cache_key = f"products:{query}:{max_price}:{product_type}"
    # cached_result = redis_client.get(cache_key)
    # if cached_result:
    #     return cached_result.decode("utf-8")

    logger.info(f"Querying products for user with query: {query}")

    try:
        query_embedding = embed_model.embed_query(query)
        index = pc.Index("product-index")
        response = index.query(
            vector=query_embedding,
            top_k=10,
            include_values=False,
            include_metadata=True,
        )

        products = []
        for match in response["matches"]:
            metadata = match["metadata"]
            product = {
                "name": metadata.get("product_name"),
                "brand": metadata.get("brand_name"),
                "category": metadata.get("category"),
                "thc": metadata.get("percentage_thc"),
                "cbd": metadata.get("percentage_cbd"),
                "price": metadata.get("latest_price"),
                "strain": metadata.get("strain"),
                "retailer_id": metadata.get("retailer_id"),
                "image_url": metadata.get("image_url"),
            }

            if (
                max_price is not None
                and product["price"] is not None
                and float(product["price"]) > max_price
            ):
                continue
            if (
                product_type
                and product["category"]
                and product["category"].lower() != product_type.lower()
            ):
                continue

            products.append(product)

        logger.debug(f"Products found: {len(products)}")

        formatted_products = [format_product_info(product) for product in products]

        response_data = {
            "query": query,
            "products": formatted_products,
            "total_results": len(formatted_products),
        }

        result = json.dumps(response_data, indent=2, cls=FirestoreEncoder)
        redis_client.set(cache_key, result, ex=3600)
        return result

    except Exception as e:
        logger.exception(f"Error querying products: {e}")
        return json.dumps(
            {
                "error": "We encountered an error while processing your request. Please try again later.",
                "query": query,
                "products": [],
                "total_results": 0,
            },
            cls=FirestoreEncoder,
        )


products_output_parser = PydanticOutputParser(pydantic_object=List[Product])
product_recommendation_tool = Tool(
    name="ProductRecommendation",
    func=get_products_from_db,
    output_parser=products_output_parser,
    description=(
        "Use this tool to retrieve cannabis products based on user queries. "
        "Provides detailed product information including product names, strain names, THC percentages (when available), "
        "CBD percentages, price ranges, and product types. Use when users ask for product details, recommendations, "
        "or specific product attributes. The tool handles cases where certain data might be unavailable."
    ),
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
        return cached_image.decode("utf-8")

    try:
        logger.info(f"Generating image with DALL-E for prompt: {prompt}")

        client = OpenAI2(api_key=os.getenv("OPENAI_API_KEY"))
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
        return cached_image.decode("utf-8")

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
            "Api-Key": f"{os.getenv('IDEOGRAM_API_KEY')}",
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

system_message = SystemMessagePromptTemplate.from_template(
    """You are an AI assistant specializing in cannabis marketing and compliance. 
    Your role is to provide accurate, helpful, and compliant information about cannabis products, 
    marketing strategies, and regulations. Always prioritize legal compliance and responsible use. 
    Use the tools provided to access specific information and generate responses."""
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

agent = create_openai_functions_agent(llm, tools, chat_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
