from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
import logging
from llama_index.llms.openai import OpenAI
import json
from .firebase_utils import db
import re

logger = logging.getLogger(__name__)
from llama_index.core.llms import ChatMessage

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o-mini")
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
    logger.debug(f"Query engine initialized for path: {path}")
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
    logger.debug(f"Generating campaign planner for template: {template_name}")
    return f"<h2>Campaign Planner Template: {template_name}</h2><p>[Template content specific to {template_name}]</p>"


def calculate_roi(investment: float, revenue: float) -> float:
    logger.debug(f"Calculating ROI for investment: {investment}, revenue: {revenue}")
    roi = (revenue - investment) / investment
    logger.debug(f"Calculated ROI: {roi}")
    return roi


def generate_compliance_checklist(state: str) -> str:
    logger.debug(f"Generating compliance checklist for state: {state}")
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
            sender="system",
            message="You are an expert in cannabis marketing, providing detailed and personalized recommendations.",
        ),
        ChatMessage(sender="user", message=question),
    ]
    logger.debug(f"Recommending cannabis strain based on question: {question}")
    resp = llm.chat(messages)
    logger.debug(f"Received recommendation response: {resp.message.content}")
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


def get_products_from_db(query: str) -> str:
    """
    Retrieve product information from the Firebase database based on the user's query.

    Args:
        query (str): A string containing the user's product query or recommendation request.

    Returns:
        str: A JSON string containing the product information or recommendations.
    """
    logger.info(f"Querying products for user with query: {query}")

    try:
        # Parse the query to extract relevant information
        keywords = query.lower().split()

        # Initialize the query
        products_ref = db.collection("products")
        query_ref = products_ref

        # Apply filters based on keywords
        if "thc" in keywords:
            query_ref = query_ref.where("data.percentage_thc", ">", 0)
        if "cbd" in keywords:
            query_ref = query_ref.where("data.percentage_cbd", ">", 0)
        if "edible" in keywords:
            query_ref = query_ref.where("data.category", "==", "Edibles")
        if "flower" in keywords:
            query_ref = query_ref.where("data.category", "==", "Flower")

        # Check for specific product or strain names
        product_name = next((word for word in keywords if len(word) > 3), None)
        if product_name:
            query_ref = query_ref.where("data.name", "==", product_name.lower())

        logger.debug(f"Query reference: {query_ref}")

        # Check for price range
        price_keywords = ["under", "over", "between"]
        price_filter = next((word for word in keywords if word in price_keywords), None)
        if price_filter:
            price_index = keywords.index(price_filter)
            if price_filter == "under" and price_index + 1 < len(keywords):
                max_price = float(re.sub(r"[^\d.]", "", keywords[price_index + 1]))
                query_ref = query_ref.where("latest_price", "<=", 30)
            elif price_filter == "over" and price_index + 1 < len(keywords):
                min_price = float(re.sub(r"[^\d.]", "", keywords[price_index + 1]))
                query_ref = query_ref.where("data.latest_price", ">=", min_price)
            elif price_filter == "between" and price_index + 2 < len(keywords):
                min_price = float(re.sub(r"[^\d.]", "", keywords[price_index + 1]))
                max_price = float(re.sub(r"[^\d.]", "", keywords[price_index + 2]))
                query_ref = query_ref.where("data.latest_price", ">=", min_price).where(
                    "data.latest_price", "<=", max_price
                )
        query_ref = query_ref.where("latest_price", "<=", 30)
        # Execute the query
        results = query_ref.limit(10).stream()  # Limit to 10 results for performance

        # Process the results
        products = []
        for doc in results:
            product_data = doc.to_dict().get("data", {})
            products.append(
                {
                    "name": product_data.get("name"),
                    "brand": product_data.get("brand_name"),
                    "category": product_data.get("category"),
                    "thc": product_data.get("percentage_thc"),
                    "cbd": product_data.get("percentage_cbd"),
                    "price": product_data.get("latest_price"),
                    "strain": product_data.get("strain"),
                }
            )

        logger.debug(f"Products found: {len(products)}")
        # Create a response
        response = {
            "query": query,
            "products": products,
            "total_results": len(products),
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error querying products: {e}")
        error_message = str(e)
        if "The query requires an index" in error_message:
            return json.dumps(
                {
                    "error": "The product search is currently unavailable due to a database configuration issue. Our team has been notified and is working on resolving it. Please try a simpler search or try again later.",
                    "query": query,
                    "products": [],
                    "total_results": 0,
                }
            )
        else:
            return json.dumps(
                {
                    "error": "An unexpected error occurred while searching for products. Please try again later.",
                    "query": query,
                    "products": [],
                    "total_results": 0,
                }
            )


# Update the FunctionTool description
product_recommendation_tool = FunctionTool.from_defaults(
    fn=get_products_from_db,
    name="ProductRecommendation",
    description="Recommends cannabis products based on user query, including product names, strain names, and price ranges. Can also find products matching strains recommended by the CannabisStrainRecommendation tool.",
)

# Combine all tools
tools = [
    compliance_guidelines_tool,
    marketing_strategies_tool,
    recommend_cannabis_strain_tool,
    seasonal_marketing_tool,
    state_policies_tool,
    # roi_calculator_tool,
    campaign_planner_tool,
    compliance_checklist_tool,
    product_recommendation_tool,
]

system_prompt = """
You are an AI-powered chatbot specialized in assisting cannabis marketers with strategy, compliance, and campaign planning.
Your main objectives are to provide accurate, up-to-date information on cannabis marketing regulations across different US states and Canada,
offer strategic marketing advice tailored to the user's experience level, assist in campaign planning including seasonal and holiday-specific strategies,
and ensure all advice adheres to legal and ethical standards in cannabis marketing.

Additionally, you have the ability to recommend cannabis strains based on specific user-provided attributes such as type, rating, desired effects, and preferred flavors. 
The personalized strain recommendation tool helps users find the best cannabis strains that match their preferences and needs.

"""

agent = OpenAIAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
)
