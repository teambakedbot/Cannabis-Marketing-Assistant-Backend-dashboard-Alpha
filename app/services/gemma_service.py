import os

os.environ["KERAS_BACKEND"] = (
    "tensorflow"  # Set TensorFlow as backend before importing keras
)

import keras
import keras_nlp
import tensorflow as tf
import tensorflow_text  # Required for GemmaTokenizer
from typing import Optional
from fastapi import HTTPException
from ..config.config import logger


class GemmaChatService:
    def __init__(self):
        try:
            # Initialize model from Hugging Face
            self.model_name = "aznatkoiny/GemmaLM-for-Cannabis"

            # Define model path
            model_path = os.path.join(
                "app", "models", "pretrained", "gemma_lm_model.keras"
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. Please download it first."
                )

            # Load the model using Keras with TensorFlow backend
            self.model = tf.keras.models.load_model(model_path)

            # Set up the sampler as specified in the model card
            self.sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
            self.model.compile(sampler=self.sampler)

            logger.info(f"Successfully loaded {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize GemmaChatService: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize cannabis language model: {str(e)}",
            )

    async def generate_response(
        self, prompt: str, max_length: Optional[int] = 256
    ) -> str:
        try:
            # Format the prompt according to the model card example
            formatted_prompt = f"Instruction:\n{prompt}\nResponse:\n"

            # Generate response using the model's native generate method
            response = self.model.generate(formatted_prompt, max_length=max_length)

            # Handle response based on its type
            if isinstance(response, list):
                # If response is a list, take the first item
                decoded_response = response[0]
            elif hasattr(response, "numpy"):
                # If response is a tensor, convert to numpy and decode
                decoded_response = response.numpy()[0].decode("utf-8")
            else:
                # If response is already a string
                decoded_response = str(response)

            # Clean up the response
            cleaned_response = decoded_response.replace(formatted_prompt, "").strip()

            return cleaned_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}",
            )
