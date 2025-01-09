from typing import List, Dict, Any, Optional, TypedDict
from functools import lru_cache
from pydantic import BaseModel, Field

# Core components
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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


class Product(BaseModel):
    """Product information."""

    id: str = Field(default="")
    cann_sku_id: str = Field(default="")
    brand_name: str = Field(default="")
    brand_id: Optional[int] = None
    url: Optional[str] = None
    image_url: str = Field(default="")
    product_name: str = Field(default="")
    category: str = Field(default="")
    subcategory: Optional[str] = None
    percentage_thc: float = Field(default=0.0)
    percentage_cbd: float = Field(default=0.0)
    latest_price: float = Field(default=0.0)
    retailer_id: str = Field(default="")
    meta_sku: str = Field(default="")


class Retailer(BaseModel):
    """Retailer information."""

    id: str = Field(default="")
    retailer_id: str = Field(default="")
    name: str = Field(default="")
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    license_number: Optional[str] = None
    serves_recreational_users: Optional[bool] = None
    serves_medical_users: Optional[bool] = None


class AgentResponse(BaseModel):
    """Structured response from the agent."""

    response: str = Field(description="The text response from the agent")
    products: List[Product] = Field(
        default_factory=list, description="List of relevant products"
    )
    retailers: List[Retailer] = Field(
        default_factory=list, description="List of relevant retailers"
    )


# Initialize the output parser
output_parser = JsonOutputParser(pydantic_object=AgentResponse)


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
            "Always prioritize legal compliance and responsible use. "
            "When you receive a tool response, DO NOT generate an additional response. "
            "The tool response already contains the properly formatted information. "
            "Your responses must be structured according to the following schema:\n"
            "{\n"
            "  'response': 'Your main text response here',\n"
            "  'products': [], // List of relevant products if any\n"
            "  'retailers': [] // List of relevant retailers if any\n"
            "}\n"
        )

        # Create the React agent with state modification
        agent_executor = create_react_agent(
            model=llm,  # Use the base model
            tools=self.tool_executor,
            state_modifier=system_prompt,
            checkpointer=self.checkpointer,
            debug=settings.DEBUG,
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
                if isinstance(
                    msg, (SystemMessage, HumanMessage, AIMessage, ToolMessage)
                ):
                    messages.append(msg)
                else:
                    # Convert dict to appropriate message type
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif role == "tool":
                        messages.append(
                            ToolMessage(content=content, name=msg.get("name", ""))
                        )

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

                # Process the result
                try:
                    # First check for tool messages
                    tool_messages = [
                        msg
                        for msg in result["messages"]
                        if isinstance(msg, ToolMessage)
                    ]
                    if tool_messages:
                        # Just return the last tool message content directly
                        return {
                            "messages": [
                                {
                                    "role": "tool",
                                    "content": tool_messages[-1].content,
                                    "name": tool_messages[-1].name,
                                }
                            ],
                            "metadata": state.get("metadata", {}),
                            "next_step": "end",
                        }

                    # If no tool message, look for AI message
                    ai_messages = [
                        msg for msg in result["messages"] if isinstance(msg, AIMessage)
                    ]
                    if ai_messages:
                        return {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": ai_messages[-1].content,
                                }
                            ],
                            "metadata": state.get("metadata", {}),
                            "next_step": "end",
                        }

                    raise ValueError("No valid response message found")

                except Exception as e:
                    logger.error(f"Error processing agent response: {e}")
                    return {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "I apologize, but I encountered an error processing the response.",
                            }
                        ],
                        "metadata": state.get("metadata", {}),
                        "next_step": "end",
                    }

            except Exception as e:
                if "maximum recursion depth exceeded" in str(e):
                    logger.warning("Agent stopped due to max iterations")
                    return {
                        "messages": [
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
                "messages": [
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
                name="GenerateImageWithDALLE",
                func=self._generate_dalle_image,
                coroutine=self._generate_dalle_image,
                args_schema=self.ImagePrompt,
                description=(
                    "Generates an image using DALL-E based on a text description. "
                    "Use this tool when a user requests an image to be created or visualized."
                ),
            ),
            Tool(
                name="GenerateImageWithIdeogram",
                func=self._generate_ideogram_image,
                coroutine=self._generate_ideogram_image,
                args_schema=self.ImagePrompt,
                description=(
                    "Generates a photorealistic image using Ideogram based on a text description. "
                    "Use this tool when a user requests a realistic image to be created or visualized."
                ),
            ),
        ]

    async def _get_retailer_info(self, query: str) -> str:
        """Get retailer information from the database."""
        try:
            query_embedding = get_text_embedding(query)
            index = pc.Index("retailer-index")
            response = index.query(
                vector=query_embedding,
                top_k=1,
                include_values=False,
                include_metadata=True,
            )

            if not response["matches"]:
                return AgentResponse(
                    response=f"No retailer found matching: {query}",
                    products=[],
                    retailers=[],
                ).model_dump_json()

            match = response["matches"][0]
            retailer_data = match.metadata

            # Convert string values to proper booleans
            serves_recreational = retailer_data.get(
                "serves_recreational_users", ""
            ).lower()
            serves_medical = retailer_data.get("serves_medical_users", "").lower()

            retailer = Retailer(
                id=match.id,
                retailer_id=retailer_data.get("retailer_id", ""),
                name=retailer_data.get("retailer_name", ""),
                address=retailer_data.get("address", ""),
                city=retailer_data.get("city", ""),
                state=retailer_data.get("state", ""),
                zip_code=retailer_data.get("zip_code", ""),
                phone=retailer_data.get("phone", ""),
                email=retailer_data.get("email", ""),
                website=retailer_data.get("website", ""),
                license_number=retailer_data.get("license_number", ""),
                serves_recreational_users=serves_recreational == "true"
                or serves_recreational == "1",
                serves_medical_users=serves_medical == "true" or serves_medical == "1",
            )

            return AgentResponse(
                response="Here is the retailer information:",
                retailers=[retailer],
                products=[],
            ).model_dump_json()

        except Exception as e:
            logger.error(f"Error getting retailer info: {e}")
            return AgentResponse(
                response=f"Error retrieving retailer information: {str(e)}",
                products=[],
                retailers=[],
            ).model_dump_json()

    async def _get_compliance_guidelines(self, query: str) -> str:
        """Get compliance guidelines based on the query."""
        try:
            query_embedding = get_text_embedding(query)
            index = pc.Index("knowledge-index")
            response = index.query(
                vector=query_embedding,
                top_k=3,
                include_values=False,
                include_metadata=True,
                namespace="Compliance guidelines",
            )

            if not response["matches"]:
                return AgentResponse(
                    response="No compliance guidelines found for your query.",
                    products=[],
                    retailers=[],
                ).model_dump_json()

            guidelines = []
            for match in response["matches"]:
                text = match["metadata"].get("text", "")
                if text:
                    guidelines.append(text)

            combined_response = "\n\n".join(guidelines)
            return AgentResponse(
                response=add_disclaimer(combined_response, "legal"),
                products=[],
                retailers=[],
            ).model_dump_json()

        except Exception as e:
            logger.error(f"Error getting compliance guidelines: {e}")
            return AgentResponse(
                response=f"Error retrieving compliance guidelines: {str(e)}",
                products=[],
                retailers=[],
            ).model_dump_json()

    async def _get_marketing_strategies(self, query: str) -> str:
        """Get marketing strategies based on the query."""
        try:
            query_embedding = get_text_embedding(query)
            index = pc.Index("knowledge-index")
            response = index.query(
                vector=query_embedding,
                top_k=3,
                include_values=False,
                include_metadata=True,
                namespace="Marketing strategies and best practices",
            )

            if not response["matches"]:
                return AgentResponse(
                    response="No marketing strategies found for your query.",
                    products=[],
                    retailers=[],
                ).model_dump_json()

            strategies = []
            for match in response["matches"]:
                text = match["metadata"].get("text", "")
                if text:
                    strategies.append(text)

            combined_response = "\n\n".join(strategies)
            return AgentResponse(
                response=add_disclaimer(combined_response, "general"),
                products=[],
                retailers=[],
            ).model_dump_json()

        except Exception as e:
            logger.error(f"Error getting marketing strategies: {e}")
            return AgentResponse(
                response=f"Error retrieving marketing strategies: {str(e)}",
                products=[],
                retailers=[],
            ).model_dump_json()

    async def _get_product_recommendations(self, query: str) -> str:
        """Get product recommendations based on search query."""
        try:
            embedding = get_text_embedding(query)
            results = index.query(
                vector=embedding,
                top_k=10,
                include_values=False,
                include_metadata=True,
            )
            if not results.matches:
                return AgentResponse(
                    response="No matching products found.", products=[], retailers=[]
                ).model_dump_json()

            products = []
            for match in results.matches:
                if match.metadata:
                    try:
                        metadata = match.metadata
                        product_data = {
                            "id": match.id,
                            "cann_sku_id": metadata.get("cann_sku_id", ""),
                            "product_name": metadata.get("product_name", ""),
                            "brand_name": metadata.get("brand_name", ""),
                            "category": metadata.get("category", ""),
                            "raw_product_category": metadata.get(
                                "raw_product_category", ""
                            ),
                            "image_url": metadata.get("image_url", ""),
                            "latest_price": float(metadata.get("latest_price", 0)),
                            "display_weight": metadata.get("display_weight", ""),
                            "percentage_thc": float(
                                metadata.get("percentage_thc", 0) or 0
                            ),
                            "percentage_cbd": float(
                                metadata.get("percentage_cbd", 0) or 0
                            ),
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
                        products.append(recommended_product)
                    except Exception as e:
                        logger.error(f"Error processing product data: {str(e)}")
                        continue

            if not products:
                return AgentResponse(
                    response="Found products but encountered errors processing the data. Please try a different search.",
                    products=[],
                    retailers=[],
                ).model_dump_json()

            return AgentResponse(
                response=f"Found {len(products)} products matching your query.",
                products=products,
                retailers=[],
            ).model_dump_json()

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return AgentResponse(
                response=f"Error retrieving product recommendations: {str(e)}",
                products=[],
                retailers=[],
            ).model_dump_json()

    async def _generate_dalle_image(self, prompt: str) -> str:
        """Generate an dalle image based on the prompt."""
        try:
            user_id = self.config.get("configurable", {}).get("user_id")
            if not user_id:
                return AgentResponse(
                    response="Image generation is only available for authenticated users.",
                    products=[],
                    retailers=[],
                ).model_dump_json()

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
            logger.info("Image generated successfully with DALL-E")

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

            return AgentResponse(
                response=f"Here's your generated image:\n\n![Generated Image]({blob.public_url})\n\n*Generated using DALL-E 3*",
                products=[],
                retailers=[],
                image_url=blob.public_url,
            ).model_dump_json()

        except Exception as e:
            logger.error(f"Error generating image with DALL-E: {e}")
            return AgentResponse(
                response=f"Error generating image: {str(e)}", products=[], retailers=[]
            ).model_dump_json()

    async def _generate_ideogram_image(self, prompt: str) -> str:
        """Generate an image using Ideogram based on the prompt."""
        try:
            user_id = self.config.get("configurable", {}).get("user_id")
            if not user_id:
                return AgentResponse(
                    response="Image generation is only available for authenticated users.",
                    products=[],
                    retailers=[],
                ).model_dump_json()

            # Generate image with Ideogram
            url = "https://api.ideogram.ai/generate"
            payload = json.dumps(
                {
                    "image_request": {
                        "prompt": prompt,
                        "style": "photo",
                        "aspectRatio": "1:1",
                        "magic_prompt_option": "AUTO",
                    }
                }
            )
            headers = {
                "Content-Type": "application/json",
                "Api-Key": settings.IDEOGRAM_API_KEY,
            }

            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()

            data = response.json()
            image_url = data["data"][0]["url"]
            logger.info("Image generated successfully with Ideogram")

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

            return AgentResponse(
                response=f"Here's your generated image:\n\n![Generated Image]({blob.public_url})\n\n*Generated using Ideogram*",
                products=[],
                retailers=[],
                image_url=blob.public_url,
            ).model_dump_json()

        except Exception as e:
            logger.error(f"Error generating image with Ideogram: {e}")
            return AgentResponse(
                response=f"Error generating image: {str(e)}", products=[], retailers=[]
            ).model_dump_json()


# Create the configurable agent instance
configurable_agent = ConfigurableAgent()

# Export the agent and llm
__all__ = ["configurable_agent", "llm"]
