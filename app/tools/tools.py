from typing import List, Dict, Any, Optional, TypedDict
from functools import lru_cache
from pydantic import BaseModel

# Core components
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# Main langchain components
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Project-specific imports
from ..config.config import settings, logger
from ..utils.redis_config import FirestoreEncoder
from ..models.schemas import Product
from ..services.firestore_chat_history import FirestoreChatMessageHistory

# External services
from openai import OpenAI
from pinecone import Pinecone
from google.cloud import storage
from firebase_admin import storage as firebase_storage

# Standard library
from typing import Dict, Any, Optional, List, TypedDict
from functools import lru_cache
import uuid
import json
import time
import tempfile
import os
import requests
from langchain.tools import Tool

# Initialize services
llm = ChatOpenAI(
    model_name=settings.OPENAI_MODEL_NAME, temperature=0.1, max_tokens=4096
)

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("product-index")


# Helper functions
def get_text_embedding(text: str) -> list:
    """Generate text embedding using OpenAI's API."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding


def add_disclaimer(response: str, disclaimer_type: str = "general") -> str:
    """Add appropriate disclaimers to responses."""
    disclaimers = {
        "legal": "\n\n*Please note: This information is provided for general informational purposes only and should not be considered legal advice.*",
        "medical": "\n\n*Please note: This information is provided for general informational purposes only and should not be considered medical advice. Consult a healthcare professional for medical concerns.*",
        "general": "\n\n*Please note: This information is provided for general informational purposes only.*",
    }
    return response + disclaimers.get(disclaimer_type, disclaimers["general"])


# Define the state schema for the chat graph
class MessagesState(TypedDict):
    """State definition for the chat graph."""

    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]
    next_step: str
    agent_scratchpad: List[Dict[str, Any]]


class ConfigurableAgent:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.checkpointer = MemorySaver()
        self.tools = self._create_tools()
        self.tool_executor = ToolExecutor(self.tools)
        self.memory = ConversationBufferMemory(
            memory_key="messages", return_messages=True, output_key="output"
        )
        self.max_iterations = 10
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Create the workflow graph with persistence."""
        # Create system prompt as state modifier
        system_prompt = (
            "You are a helpful assistant specializing in cannabis marketing. "
            "Answer all questions to the best of your ability using the available tools. "
            "Keep responses concise unless asked for more detail. "
            "Always prioritize legal compliance and responsible use."
        )

        # Create the React agent with state modification
        agent_executor = create_react_agent(
            model=llm,  # The LangChain chat model
            tools=self.tool_executor,  # Pass the ToolExecutor instance
            state_modifier=system_prompt,  # System prompt as state modifier
            checkpointer=self.checkpointer,  # For persisting state
            debug=settings.DEBUG,  # Enable debug mode based on settings
        )

        return agent_executor

    async def ainvoke(
        self, state: MessagesState, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a message using the workflow."""
        try:
            if config:
                self.config.update(config.get("configurable", {}))

            # Ensure we have a thread_id
            if "thread_id" not in self.config.get("configurable", {}):
                self.config["configurable"] = {
                    **(self.config.get("configurable", {})),
                    "thread_id": str(uuid.uuid4()),
                }

            # Convert state messages to LangGraph format
            messages = []
            for msg in state["messages"]:
                if isinstance(msg, dict):
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                elif isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    if role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role in ["human", "user"]:
                        messages.append(HumanMessage(content=content))
                    elif role in ["ai", "assistant"]:
                        messages.append(AIMessage(content=content))

            # Create agent state
            agent_state = {
                "messages": messages,
                "is_last_step": False,  # Required by AgentState schema
            }

            try:
                # Execute agent with recursion limit
                result = await self.workflow.ainvoke(
                    agent_state,
                    config={
                        "recursion_limit": self.max_iterations,
                        "configurable": self.config.get("configurable", {}),
                    },
                )

                # Convert result back to expected format
                formatted_messages = []
                for msg in result["messages"]:
                    if isinstance(msg, SystemMessage):
                        formatted_messages.append(
                            {"role": "system", "content": msg.content}
                        )
                    elif isinstance(msg, HumanMessage):
                        formatted_messages.append(
                            {"role": "user", "content": msg.content}
                        )
                    elif isinstance(msg, AIMessage):
                        formatted_messages.append(
                            {"role": "assistant", "content": msg.content}
                        )
                    elif isinstance(msg, tuple) and len(msg) == 2:
                        role, content = msg
                        if role == "system":
                            formatted_messages.append(
                                {"role": "system", "content": content}
                            )
                        elif role in ["human", "user"]:
                            formatted_messages.append(
                                {"role": "user", "content": content}
                            )
                        elif role in ["ai", "assistant"]:
                            formatted_messages.append(
                                {"role": "assistant", "content": content}
                            )

                return {
                    "messages": formatted_messages,
                    "metadata": state.get("metadata", {}),
                    "next_step": "end",
                }

            except Exception as e:
                if "maximum recursion depth exceeded" in str(e):
                    logger.warning("Agent stopped due to max iterations")
                    return {
                        "messages": state["messages"]
                        + [
                            {
                                "role": "assistant",
                                "content": "I apologize, but I've reached my iteration limit. Let me summarize what I know so far...",
                            }
                        ],
                        "metadata": state.get("metadata", {}),
                        "next_step": "end",
                    }
                raise

        except Exception as e:
            logger.error(f"Error in ainvoke: {str(e)}")
            return {
                "messages": state["messages"]
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
                    "role": "ai",
                    "content": "I apologize, but I encountered an error. Please try again.",
                }
            ],
        }

    # Tool schemas
    class SearchQuery(BaseModel):
        """Schema for search query tools."""

        query: str = "Search query string"

    class ImagePrompt(BaseModel):
        """Schema for image generation tool."""

        prompt: str = "Image description"

    def _create_tools(self):
        """Create the tools available to the agent."""
        return [
            Tool(
                name="ProductRecommendation",
                description="Get product recommendations based on search query",
                func=self._get_product_recommendations,
                coroutine=self._get_product_recommendations,
                args_schema=self.SearchQuery,
            ),
            Tool(
                name="RetailerInformation",
                func=self._get_retailer_info,
                coroutine=self._get_retailer_info,
                args_schema=self.SearchQuery,
                description=(
                    "Use this tool to find detailed information about cannabis retailers. "
                    "Provides location details, contact information, operating hours, and services offered. "
                    "Can search by name, location, or other criteria."
                ),
            ),
            Tool(
                name="ComplianceGuidelines",
                func=self._get_compliance_guidelines,
                coroutine=self._get_compliance_guidelines,
                args_schema=self.SearchQuery,
                description=(
                    "Use this tool for questions about cannabis marketing compliance requirements, "
                    "regulations, and guidelines. Provides detailed compliance information with citations."
                ),
            ),
            Tool(
                name="MarketingStrategies",
                func=self._get_marketing_strategies,
                coroutine=self._get_marketing_strategies,
                args_schema=self.SearchQuery,
                description=(
                    "Use this tool for questions about effective cannabis marketing strategies, "
                    "best practices, and campaign ideas. Provides actionable marketing advice "
                    "that is compliant with regulations."
                ),
            ),
            Tool(
                name="GenerateImage",
                func=self._generate_image,
                coroutine=self._generate_image,
                args_schema=self.ImagePrompt,
                description=(
                    "Generates an image using AI based on a text description. "
                    "Use this tool when a user requests an image to be created or visualized."
                ),
            ),
        ]

    async def _get_retailer_info(self, query: str) -> str:
        """Get retailer information from the database."""
        try:
            # Get the embedding for the query
            query_embedding = get_text_embedding(query)

            # Query the retailer index
            index = pc.Index("retailer-index")
            response = index.query(
                vector=query_embedding,
                top_k=1,
                include_values=False,
                include_metadata=True,
            )

            if not response["matches"]:
                return f"No retailer found matching: {query}"

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
                "serves_recreational_users": retailer_data.get(
                    "serves_recreational_users"
                ),
                "serves_medical_users": retailer_data.get("serves_medical_users"),
                "latitude": retailer_data.get("latitude"),
                "longitude": retailer_data.get("longitude"),
            }
            return json.dumps(response_data, indent=2, cls=FirestoreEncoder)

        except Exception as e:
            logger.error(f"Error getting retailer info: {e}")
            return f"Error retrieving retailer information: {str(e)}"

    async def _get_compliance_guidelines(self, query: str) -> str:
        """Get compliance guidelines based on the query."""
        try:
            # Get the embedding for the query
            query_embedding = get_text_embedding(query)

            # Query the compliance guidelines index
            index = pc.Index("knowledge-index")
            response = index.query(
                vector=query_embedding,
                top_k=3,
                include_values=False,
                include_metadata=True,
                namespace="Compliance guidelines",
            )

            if not response["matches"]:
                return "No compliance guidelines found for your query."

            # Combine the relevant guidelines
            guidelines = []
            for match in response["matches"]:
                text = match["metadata"].get("text", "")
                if text:
                    guidelines.append(text)

            combined_response = "\n\n".join(guidelines)
            return add_disclaimer(combined_response, "legal")

        except Exception as e:
            logger.error(f"Error getting compliance guidelines: {e}")
            return f"Error retrieving compliance guidelines: {str(e)}"

    async def _get_marketing_strategies(self, query: str) -> str:
        """Get marketing strategies based on the query."""
        try:
            # Get the embedding for the query
            query_embedding = get_text_embedding(query)

            # Query the marketing strategies index
            index = pc.Index("knowledge-index")
            response = index.query(
                vector=query_embedding,
                top_k=3,
                include_values=False,
                include_metadata=True,
                namespace="Marketing strategies and best practices",
            )

            if not response["matches"]:
                return "No marketing strategies found for your query."

            # Combine the relevant strategies
            strategies = []
            for match in response["matches"]:
                text = match["metadata"].get("text", "")
                if text:
                    strategies.append(text)

            combined_response = "\n\n".join(strategies)
            return add_disclaimer(combined_response, "general")

        except Exception as e:
            logger.error(f"Error getting marketing strategies: {e}")
            return f"Error retrieving marketing strategies: {str(e)}"

    async def _get_product_recommendations(self, query: str) -> str:
        """Get product recommendations based on search query."""
        try:
            # Convert query to embedding
            embedding = get_text_embedding(query)

            # Search Pinecone
            results = index.query(vector=embedding, top_k=5, include_metadata=True)

            if not results.matches:
                return "No matching products found."

            # Group products by meta_sku
            products_by_meta_sku = {}
            for match in results.matches:
                if match.metadata:
                    try:
                        meta_sku = match.metadata.get("meta_sku")
                        if not meta_sku:
                            continue

                        # Convert metadata to Product format
                        product = {
                            "cann_sku_id": match.metadata.get("cann_sku_id", ""),
                            "brand_name": match.metadata.get("brand_name", ""),
                            "brand_id": (
                                int(match.metadata.get("brand_id"))
                                if match.metadata.get("brand_id")
                                else None
                            ),
                            "url": match.metadata.get("url"),
                            "image_url": match.metadata.get("image_url", ""),
                            "raw_product_name": match.metadata.get(
                                "raw_product_name", ""
                            ),
                            "product_name": match.metadata.get("product_name", ""),
                            "raw_weight_string": match.metadata.get(
                                "raw_weight_string"
                            ),
                            "display_weight": match.metadata.get("display_weight"),
                            "raw_product_category": match.metadata.get(
                                "raw_product_category"
                            ),
                            "category": match.metadata.get("category", ""),
                            "raw_subcategory": match.metadata.get("raw_subcategory"),
                            "subcategory": match.metadata.get("subcategory"),
                            "product_tags": (
                                eval(match.metadata.get("product_tags", "[]"))
                                if match.metadata.get("product_tags")
                                else None
                            ),
                            "percentage_thc": (
                                float(match.metadata.get("percentage_thc", 0))
                                if match.metadata.get("percentage_thc")
                                else 0.0
                            ),
                            "percentage_cbd": (
                                float(match.metadata.get("percentage_cbd", 0))
                                if match.metadata.get("percentage_cbd")
                                else 0.0
                            ),
                            "mg_thc": (
                                float(match.metadata.get("mg_thc", 0))
                                if match.metadata.get("mg_thc")
                                else 0.0
                            ),
                            "mg_cbd": (
                                float(match.metadata.get("mg_cbd", 0))
                                if match.metadata.get("mg_cbd")
                                else 0.0
                            ),
                            "quantity_per_package": (
                                int(match.metadata.get("quantity_per_package"))
                                if match.metadata.get("quantity_per_package")
                                else None
                            ),
                            "medical": match.metadata.get("medical", "False").lower()
                            == "true",
                            "recreational": match.metadata.get(
                                "recreational", "False"
                            ).lower()
                            == "true",
                            "latest_price": float(
                                match.metadata.get("latest_price", 0)
                            ),
                            "menu_provider": match.metadata.get("menu_provider", ""),
                            "retailer_id": match.metadata.get("retailer_id", ""),
                            "meta_sku": meta_sku,
                            "id": match.id,
                        }

                        if meta_sku not in products_by_meta_sku:
                            products_by_meta_sku[meta_sku] = {
                                "meta_sku": meta_sku,
                                "retailer_id": product["retailer_id"],
                                "products": [],
                            }
                        products_by_meta_sku[meta_sku]["products"].append(product)
                    except Exception as e:
                        logger.error(f"Error processing product data: {str(e)}")
                        continue

            if not products_by_meta_sku:
                return "Found products but encountered errors processing the data. Please try a different search."

            # Create summary message
            total_products = sum(
                len(group["products"]) for group in products_by_meta_sku.values()
            )
            summary_message = f"Found {total_products} products matching your query."
            logger.debug(f"Created summary message: {summary_message}")

            # Format product data
            product_data = {"products": list(products_by_meta_sku.values())}
            logger.debug(
                f"Formatted product data: {json.dumps(product_data, indent=2)}"
            )

            # Return JSON-encoded string containing list of summary message and product data
            return json.dumps([summary_message, product_data])

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return f"Error retrieving product recommendations: {str(e)}"

    async def _generate_image(self, prompt: str) -> str:
        """Generate an image based on the prompt."""
        try:
            user_id = self.config.get("configurable", {}).get("user_id")
            if not user_id:
                return "Image generation is only available for authenticated users."

            # Generate image with DALL-E
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
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
            bucket = storage.bucket(name=settings.FIREBASE_STORAGE_BUCKET)
            timestamp = int(time.time())
            filename = "".join(
                x if x.isalnum() or x in ("_", "-") else "_" for x in prompt
            )[:50]
            blob_path = f"temp_images/{user_id}/{filename}_{timestamp}.png"

            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()

            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

            # Upload to Firebase Storage
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(temp_path)
            blob.make_public()

            # Clean up temp file
            os.unlink(temp_path)

            return f"Here's your generated image:\n\n![Generated Image]({blob.public_url})\n\n*Generated using DALL-E 3*"

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return f"Error generating image: {str(e)}"


# Create the configurable agent instance
configurable_agent = ConfigurableAgent()

# Export the agent and llm
__all__ = ["configurable_agent", "llm"]
