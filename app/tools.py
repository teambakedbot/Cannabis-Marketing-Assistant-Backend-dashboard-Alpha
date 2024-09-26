from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    ServiceContext,
    GPTVectorStoreIndex,
)
import logging
import json
from .firebase_utils import db
import re
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from .config import settings
from pinecone import Pinecone
from openai import OpenAI as OpenAI2
import base64
import os
import requests
from .redis_config import FirestoreEncoder
from .config import logger

llm = OpenAI(model="gpt-4o-mini")

pinecone_api_key = settings.PINECONE_API_KEY
index_name = "knowledge-index"

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
pc = Pinecone(api_key=settings.PINECONE_API_KEY)


def get_compliance_guidelines(query: str) -> str:
    response = compliance_guidelines.query(query)
    return add_disclaimer(response.response, "legal")


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


# Function to get the query engine
def get_query_engine(namespace: str):
    vector_store = PineconeVectorStore(index_name=index_name, namespace=namespace)

    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # Create a VectorStoreIndex with the vector store
    index_instance = GPTVectorStoreIndex(
        [], service_context=service_context, vector_store=vector_store
    )

    # Get the query engine
    query_engine = index_instance.as_query_engine()

    return query_engine


# Initialize query engines with Pinecone
compliance_guidelines = get_query_engine("Compliance guidelines")
marketing_strategies = get_query_engine("Marketing strategies and best practices")
seasonal_marketing = get_query_engine("Seasonal and holiday marketing plans")
state_policies = get_query_engine("State-specific cannabis marketing regulations")

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


def get_state_policies(query: str) -> str:
    response = state_policies.query(query)
    return add_disclaimer(response.response, "legal")


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
    checklist = f"<h2>Compliance Checklist for {state}</h2><ul><li>Checklist item 1</li><li>Checklist item 2</li><li>Checklist item 3</li></ul>"
    return add_disclaimer(checklist, "legal")


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
def recommend_cannabis_strain(question: str) -> list:
    """
    Recommend a cannabis strain based on the given question.
    """
    messages = [
        ChatMessage(
            sender="system",
            message="You are an expert in cannabis marketing, providing detailed and personalized recommendations.",
        ),
        ChatMessage(sender="user", message=question),
    ]
    logger.debug(f"Recommending cannabis strain based on question: {question}")

    # Log the messages being sent to the LLM
    logger.debug(f"Messages sent to LLM: {messages}")

    resp = llm.chat(messages)
    logger.debug(f"Received recommendation response: {resp.message.content}")

    # Split the response into two messages
    answer_parts = resp.message.content.split("\n\n", 1)  # Split into two parts
    return (
        answer_parts if len(answer_parts) > 1 else [resp.message.content, ""]
    )  # Ensure two messages


# Create the FunctionTool with a detailed description
recommend_cannabis_strain_tool = FunctionTool.from_defaults(
    recommend_cannabis_strain,
    name="CannabisStrainRecommendation",
    description=(
        "This tool recommends a cannabis strain with details based on a detailed user question. "
    ),
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
        f"<div class='product-info'>\n"
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

    Args:
        query (str): A string containing the user's product query or recommendation request.
        max_price (float, optional): The maximum price for filtering products.
        product_type (str, optional): The specific product type for filtering.

    Returns:
        str: A JSON string containing the product information or recommendations.
    """
    logger.info(f"Querying products for user with query: {query}")

    try:
        # Generate the embedding for the query
        query_embedding = embed_model.get_text_embedding(query)
        index = pc.Index("product-index")
        # Perform search in Pinecone
        response = index.query(
            vector=query_embedding,
            top_k=10,
            include_values=False,
            include_metadata=True,
        )

        # Process the results
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

            # Apply filters
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

        # Format product information
        formatted_products = [format_product_info(product) for product in products]

        # Create a response
        response_data = {
            "query": query,
            "products": formatted_products,
            "total_results": len(formatted_products),
        }

        return json.dumps(response_data, indent=2, cls=FirestoreEncoder)

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


# Update the FunctionTool description
product_recommendation_tool = FunctionTool.from_defaults(
    fn=get_products_from_db,
    name="ProductRecommendation",
    description=(
        "Use this tool to retrieve cannabis products based on user queries. "
        "Provides detailed product information including product names, strain names, THC percentages (when available), "
        "CBD percentages, price ranges, and product types. Use when users ask for product details, recommendations, "
        "or specific product attributes. The tool handles cases where certain data might be unavailable."
    ),
)


def get_retailer_info(query: str) -> str:
    """
    Retrieve retailer information from the Pinecone vector database based on the user's query.

    Args:
        query (str): The user's query about the retailer.

    Returns:
        str: A JSON string containing the retailer information.
    """
    logger.info(f"Fetching retailer information for query: {query}")

    try:
        # Generate the embedding for the query
        query_embedding = embed_model.get_text_embedding(query)

        index = pc.Index("retailer-index")
        # Perform search in Pinecone
        response = index.query(
            vector=query_embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
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


# Update the FunctionTool for retailer information
retailer_info_tool = FunctionTool.from_defaults(
    fn=get_retailer_info,
    name="RetailerInformation",
    description="Retrieves detailed information about a cannabis retailer based on the user's query. This tool can be used to get additional context about retailers, including address and contact information.",
)

general_knowledge = get_query_engine("General Cannabis Knowledge")

general_knowledge_tool = QueryEngineTool(
    query_engine=general_knowledge,
    metadata=ToolMetadata(
        name="General_Cannabis_Knowledge",
        description="Provides general information about cannabis, including effects, usage, and terminology.",
    ),
)

usage_instructions = get_query_engine("Cannabis Usage Instructions")

usage_instructions_tool = QueryEngineTool(
    query_engine=usage_instructions,
    metadata=ToolMetadata(
        name="Usage_Instructions",
        description="Provides step-by-step instructions on how to use different cannabis products and consumption methods.",
    ),
)


def medical_information(query: str) -> str:
    """
    Provide general information about medical cannabis use.

    Args:
        query (str): The user's query about medical cannabis.

    Returns:
        str: A response with general medical information and a disclaimer.
    """
    # Implementation to provide general medical information
    # This is a placeholder and should be replaced with actual implementation
    response = f"Here is some general information about medical cannabis use related to your query: '{query}'"
    return add_disclaimer(response, "medical")


medical_information_tool = FunctionTool.from_defaults(
    fn=medical_information,
    name="MedicalInformation",
    description="Provides general information about medical cannabis. Does not offer medical advice.",
)


def generate_image_with_dalle(prompt: str) -> str:
    """
    Generate an image using DALL-E based on the given prompt.

    Args:
        prompt (str): The description of the image to generate.

    Returns:
        str: A base64-encoded string of the generated image.
    """
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

        # Extract the base64-encoded image
        image_data = response.data[0].url
        logger.info("Image generated successfully")
        return image_data
    except Exception as e:
        logger.error(f"Error generating image with DALL-E: {str(e)}")
        return ""


# Create the FunctionTool for image generation
image_generation_tool = FunctionTool.from_defaults(
    fn=generate_image_with_dalle,
    name="GenerateImageWithDALLE",
    description="Generates an image using DALL-E based on a text description. Use this tool when a user requests an image to be created or visualized.",
)


def generate_image_with_ideogram(prompt: str) -> str:
    """
    Generate an image using Ideogram based on the given prompt.

    Args:
        prompt (str): The description of the image to generate.

    Returns:
        str: The URL of the generated image.
    """
    try:
        logger.info(f"Generating image with Ideogram for prompt: {prompt}")

        url = "https://ideogram.ai/api/images/generate"

        payload = json.dumps(
            {"prompt": prompt, "style": "photo", "aspectRatio": "1:1"},
            cls=FirestoreEncoder,
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('IDEOGRAM_API_KEY')}",
        }

        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        data = response.json()
        image_url = data["images"][0]["url"]

        logger.info("Image generated successfully with Ideogram")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image with Ideogram: {str(e)}")
        return ""


# Create the FunctionTool for Ideogram image generation
ideogram_image_generation_tool = FunctionTool.from_defaults(
    fn=generate_image_with_ideogram,
    name="GenerateImageWithIdeogram",
    description="Generates a photorealistic image using Ideogram based on a text description. Use this tool when a user requests a highly detailed, photorealistic image to be created or visualized.",
)

# Combine all tools
tools = [
    compliance_guidelines_tool,
    marketing_strategies_tool,
    # recommend_cannabis_strain_tool,
    seasonal_marketing_tool,
    state_policies_tool,
    general_knowledge_tool,
    # roi_calculator_tool,
    campaign_planner_tool,
    compliance_checklist_tool,
    product_recommendation_tool,
    retailer_info_tool,
    usage_instructions_tool,
    medical_information_tool,
    image_generation_tool,
    # ideogram_image_generation_tool,
]

system_prompt = """
You are Smokey, an AI-powered chatbot specialized in assisting cannabis users and marketers with information, recommendations, and guidance.
Your main objectives are to:
- Provide accurate, up-to-date information on cannabis strains, effects, and usage.
- Offer product recommendations based on user preferences, **including specific details like THC percentages when available**.
- Assist with finding nearby dispensaries and retailer information.
- Provide compliance guidelines and marketing strategies for cannabis businesses.
- Answer general questions about cannabis laws, effects, and terminology.
- Ensure all advice adheres to legal and ethical standards in cannabis use and marketing.
- Generate images related to cannabis using either DALL-E or Ideogram, depending on the user's needs.
- Do not mention any limitations about data access. Always use the tools provided to answer queries.

When answering queries, follow these steps:
1. Analyze the user's question and identify the key information they're seeking.
2. Determine which tool(s) would be most appropriate to answer the query.
3. Use the selected tool(s) to gather the necessary information.
4. Synthesize the information into a clear, concise response.
5. If any information is unavailable, acknowledge this and provide alternative relevant details.

Here are some examples of how to handle specific types of queries:

User: "Show me products with high THC content."
Agent thought process: The user is looking for products with high THC content. I should use the ProductRecommendation tool to find relevant products and focus on those with high THC percentages.
Agent: "Certainly! I'll find some products with high THC content for you. Here are a few options:

1. **OG Kush** by Green Farms - Category: Flower
   THC: 28%, CBD: 0.5%
   Price: $45.00
   Strain: Hybrid

2. **Blue Dream Vape** by Vapor Co. - Category: Vape Cartridge
   THC: 85%, CBD: 2%
   Price: $60.00
   Strain: Hybrid

3. **Gorilla Glue #4** by Sticky Buds - Category: Flower
   THC: 26%, CBD: 1%
   Price: $50.00
   Strain: Hybrid

These products all have high THC content, ranging from 26% to 85%. Remember that THC potency can vary, and higher percentages don't necessarily mean a better experience. It's important to start low and go slow, especially if you're new to cannabis or trying a new product."

User: "What are the legal requirements for cannabis advertising in California?"
Agent thought process: This query is about legal compliance for cannabis advertising in a specific state. I should use the State_Policies tool to get accurate, up-to-date information about California's regulations.
Agent: "To answer your question about legal requirements for cannabis advertising in California, I'll consult our database of state-specific cannabis regulations. Here's what I found:

1. Age Restrictions: All cannabis advertisements must be restricted to audiences where at least 71.6% of the audience is reasonably expected to be 21 years of age or older.

2. Content Limitations: Advertisements cannot contain any content that is attractive to children, including cartoons, toys, or similar images and names that resemble candy products.

3. Health Claims: It's prohibited to make any health-related statements about cannabis products unless there's reliable scientific evidence to support such claims.

4. Location Restrictions: Cannabis advertisements are not allowed within 1,000 feet of day care centers, schools, playgrounds, or youth centers.

5. Required Warnings: All advertisements must include the state's required warning message about the health risks of cannabis use.

6. License Information: Advertisements must include the business's state license number.

7. Interstate Advertising: It's illegal to advertise California cannabis businesses or products across state lines.

Please note that these regulations can change, and local jurisdictions may have additional requirements. It's always best to consult with a legal professional for the most current and comprehensive advice on cannabis advertising compliance.

*This information is provided for general informational purposes only and should not be considered legal advice.*"

For image generation:
- Use the Ideogram tool (GenerateImageWithIdeogram) when users need highly detailed, photorealistic images of cannabis products, plants, or related scenes.

Remember to always use the tools provided to answer queries and present available product details such as THC percentages, CBD percentages, prices, and categories when users ask for product information. Do not mention any limitations about not having access to data.
"""


agent = OpenAIAgent.from_tools(
    tools=tools, llm=llm, verbose=True, system_prompt=system_prompt, max_iterations=5
)
