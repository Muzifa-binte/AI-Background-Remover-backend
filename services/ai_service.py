import os
import base64
import json
import re
import asyncio
import logging
from io import BytesIO
from typing import Optional, Callable, Any
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Guard optional AI-provider packages so that a missing package for the
# *unused* provider does not crash the backend at import time.
try:
    from openai import AsyncOpenAI
    _openai_available = True
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    _openai_available = False

try:
    import google.generativeai as genai
    _genai_available = True
except ImportError:
    genai = None  # type: ignore[assignment]
    _genai_available = False

class AIService:
    """Handles communication with either Groq or Gemini AI providers with enhanced error handling and retry logic."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_provider = None
        
        # Try to initialize primary provider
        try:
            self._initialize_provider(self.provider)
        except Exception as e:
            logger.warning(f"Failed to initialize primary provider {self.provider}: {str(e)}")
            # Try fallback provider
            self.fallback_provider = "groq" if self.provider == "gemini" else "gemini"
            try:
                logger.info(f"Attempting to initialize fallback provider: {self.fallback_provider}")
                self._initialize_provider(self.fallback_provider)
                self.provider = self.fallback_provider
                logger.info(f"Successfully switched to fallback provider: {self.provider}")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback provider {self.fallback_provider}: {str(fallback_error)}")
                # Set up as unconfigured but don't crash
                self.is_configured = False
                self.client = None

    def _initialize_provider(self, provider: str):
        """Initialize a specific AI provider."""
        if provider == "gemini":
            if not _genai_available:
                raise ImportError(
                    "AI_PROVIDER is set to 'gemini' but the 'google-generativeai' package is not "
                    "installed. Run: pip install google-generativeai"
                )
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            self.is_configured = bool(self.api_key and "your_gemini_api_key" not in self.api_key)
            if self.is_configured:
                genai.configure(api_key=self.api_key)
            # Default to gemini-3.5-flash-lite (fast, efficient, multimodal).
            # Override per-model via env vars if you want a different tier:
            #   GEMINI_CHAT_MODEL   e.g. gemini-3.6-flash or gemini-3.7-flash
            #   GEMINI_VISION_MODEL e.g. gemini-3.6-flash or gemini-3.5-flash-lite
            self.chat_model   = os.getenv("GEMINI_CHAT_MODEL",   "gemini-3.5-flash-lite")
            self.vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
            self.client = None
        else:
            if not _openai_available:
                raise ImportError(
                    "AI_PROVIDER is set to 'groq' but the 'openai' package is not installed. "
                    "Run: pip install openai"
                )
            self.api_key = os.getenv("GROQ_API_KEY", "")
            self.is_configured = bool(self.api_key and "your_groq_api_key" not in self.api_key)
            if self.is_configured:
                self.client = AsyncOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key
                )
            else:
                self.client = None
            self.chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
            self.vision_model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

    async def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Retry a function with exponential backoff."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Don't retry on certain errors
                if error_type in ['ValueError', 'ImportError']:
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise
                
                # Log retry attempt
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {func.__name__}: {str(e)}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {func.__name__}: {str(e)}")
        
        raise last_exception

    def _verify_configuration(self):
        """Verify AI configuration with helpful error messages and fallback support."""
        if not self.is_configured:
            # Try to reload environment variables
            from dotenv import load_dotenv
            from pathlib import Path
            env_path = Path(__file__).resolve().parent.parent / ".env"
            load_dotenv(dotenv_path=env_path, override=True)

            # Re-check configuration after reload
            self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
            if self.provider == "gemini":
                self.api_key = os.getenv("GEMINI_API_KEY", "")
                self.is_configured = bool(self.api_key and "your_gemini_api_key" not in self.api_key)
                if self.is_configured and _genai_available:
                    try:
                        genai.configure(api_key=self.api_key)
                    except Exception as e:
                        logger.error(f"Failed to configure Gemini: {str(e)}")
                        self.is_configured = False
                self.chat_model   = os.getenv("GEMINI_CHAT_MODEL",   "gemini-3.5-flash-lite")
                self.vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
            else:
                self.api_key = os.getenv("GROQ_API_KEY", "")
                self.is_configured = bool(self.api_key and "your_groq_api_key" not in self.api_key)
                if self.is_configured and _openai_available:
                    try:
                        self.client = AsyncOpenAI(
                            base_url="https://api.groq.com/openai/v1",
                            api_key=self.api_key
                        )
                    except Exception as e:
                        logger.error(f"Failed to configure Groq client: {str(e)}")
                        self.is_configured = False
                self.chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
                self.vision_model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

        # If still not configured, try fallback provider
        if not self.is_configured and self.fallback_provider:
            logger.info(f"Primary provider not configured, trying fallback: {self.fallback_provider}")
            try:
                self._initialize_provider(self.fallback_provider)
                self.provider = self.fallback_provider
                logger.info(f"Successfully switched to fallback provider: {self.provider}")
                return
            except Exception as e:
                logger.error(f"Fallback provider also failed: {str(e)}")

        # Final check with user-friendly error message
        if not self.is_configured:
            provider_name = "Gemini" if self.provider == "gemini" else "Groq"
            env_var = "GEMINI_API_KEY" if self.provider == "gemini" else "GROQ_API_KEY"
            
            # Check if it's a placeholder key
            if self.api_key and ("your_" in self.api_key.lower() or "placeholder" in self.api_key.lower()):
                raise ValueError(
                    f"{provider_name} API key is still set to a placeholder value. "
                    f"Please replace 'your_{env_var.lower()}' with your actual API key in the .env file. "
                    f"You can get a free API key from the {provider_name} developer console."
                )
            
            # Check if package is missing
            if self.provider == "gemini" and not _genai_available:
                raise ImportError(
                    f"The 'google-generativeai' package is not installed. "
                    f"Run: pip install google-generativeai"
                )
            elif self.provider == "groq" and not _openai_available:
                raise ImportError(
                    f"The 'openai' package is not installed. "
                    f"Run: pip install openai"
                )
            
            # Generic missing key error
            raise ValueError(
                f"{provider_name} API key is not configured or invalid. "
                f"Please set a valid {env_var} in the .env file. "
                f"Make sure the .env file exists in the backend directory and contains your API key."
            )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        return text.strip()

    def _extract_json(self, text: str):
        text = self._clean_text(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        match_braces = re.search(r'(\{.*\})', text, re.DOTALL)
        if match_braces:
            try:
                return json.loads(match_braces.group(1).strip())
            except json.JSONDecodeError:
                pass

        match_brackets = re.search(r'(\[.*\])', text, re.DOTALL)
        if match_brackets:
            try:
                return json.loads(match_brackets.group(1).strip())
            except json.JSONDecodeError:
                pass

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        fallback_list = []
        for line in lines:
            cleaned_line = re.sub(r'^(?:\d+\.|\*|-)\s*', '', line).strip().strip('"').strip("'")
            if cleaned_line and len(cleaned_line) > 3 and not cleaned_line.startswith('<'):
                fallback_list.append(cleaned_line)
        
        if fallback_list:
            return fallback_list

        raise ValueError(f"Could not parse valid JSON or list from AI response: {text}")

    def _extract_thinking(self, text: str) -> tuple[str, str | None]:
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
        else:
            open_match = re.search(r'<think>(.*)', text, re.DOTALL)
            thinking = open_match.group(1).strip() if open_match else None
        reply = self._clean_text(text)
        return reply, thinking

    async def chat(self, message: str, image_bytes: bytes = None) -> tuple[str, str | None]:
        """Chat with AI assistant with retry logic and enhanced error handling."""
        self._verify_configuration()
        
        async def _chat_impl():
            if self.provider == "gemini":
                model = genai.GenerativeModel(self.chat_model)
                if image_bytes:
                    img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    prompt = (
                        "You are a professional designer and helpful assistant for the AI Background Remover application. "
                        "You are answering user queries about the uploaded image. Help them with background recommendations, editing advice, and captions.\n\n"
                        f"User Message: {message}"
                    )
                    response = await model.generate_content_async([prompt, img])
                else:
                    response = await model.generate_content_async(message)
                
                raw_reply = response.text or "No response from AI."
                return self._extract_thinking(raw_reply)
            else:
                if image_bytes:
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                    
                    response = await self.client.chat.completions.create(
                        model=self.vision_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a professional designer and helpful assistant for the AI Background Remover application. "
                                    "You are answering user queries about the uploaded image. Help them with background recommendations, editing advice, and captions."
                                )
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": message},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1024
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.chat_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful AI Assistant for the AI Background Remover application. You help users analyze images, suggest background replacements, write creative captions, and answer design queries."},
                            {"role": "user", "content": message}
                        ],
                        temperature=0.7,
                        max_tokens=1024
                    )
                raw_reply = response.choices[0].message.content or "No response from AI."
                return self._extract_thinking(raw_reply)
        
        try:
            return await self._retry_with_backoff(_chat_impl)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chat failed after retries: {error_msg}")
            
            # Provide user-friendly error messages
            if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError("AI service authentication failed. Please check your API key configuration.")
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("AI service rate limit exceeded. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise RuntimeError("AI service timed out. Please check your connection and try again.")
            elif "quota" in error_msg.lower():
                raise RuntimeError("AI service quota exceeded. Please check your plan and usage.")
            else:
                raise RuntimeError(f"Unable to get AI response: {error_msg}. Please try again later.")

    async def analyze_image(self, image_bytes: bytes) -> dict:
        """Analyze image with retry logic and enhanced error handling."""
        self._verify_configuration()
        
        async def _analyze_impl():
            system_prompt = (
                "You are an expert design AI. Analyze this image and extract composition details. "
                "Respond ONLY with a valid JSON object matching the following structure:\n"
                '{\n'
                '  "subject": "The primary subject in the image (e.g. A man in a green coat, a cosmetic bottle, etc.)",\n'
                '  "image_type": "The category of the image (e.g. Portrait, Product Shot, Landscape, Food)",\n'
                '  "background_description": "Detailed description of the current background elements, colors, and textures",\n'
                '  "suggested_use": "Recommended marketing/design use cases for this image after background removal",\n'
                '  "editing_recommendations": [\n'
                '    "Step 1 recommendation (e.g. Feather boundaries to preserve hair detail)",\n'
                '    "Step 2 recommendation (e.g. Adjust lighting to match a studio look)",\n'
                '    "Step 3 recommendation (e.g. Add soft drop-shadow under the subject)"\n'
                '  ]\n'
                '}'
            )

            if self.provider == "gemini":
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([system_prompt, img])
                raw_text = response.text or ""
                return self._extract_json(raw_text)
            else:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"

                response = await self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this image and return the composition details as a JSON object."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024
                )
                raw_text = response.choices[0].message.content or ""
                return self._extract_json(raw_text)
        
        try:
            return await self._retry_with_backoff(_analyze_impl)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Image analysis failed after retries: {error_msg}")
            
            if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError("AI service authentication failed during image analysis. Please check your API key configuration.")
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("AI service rate limit exceeded during image analysis. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise RuntimeError("AI service timed out during image analysis. Please check your connection and try again.")
            elif "quota" in error_msg.lower():
                raise RuntimeError("AI service quota exceeded during image analysis. Please check your plan and usage.")
            else:
                raise RuntimeError(f"Unable to analyze image: {error_msg}. Please try again later.")

    async def generate_caption(self, image_bytes: bytes, style: str = "casual") -> str:
        """Generate single caption with retry logic and enhanced error handling."""
        self._verify_configuration()
        
        async def _caption_impl():
            style_guide = {
                "instagram": "fun, engaging, casual with relevant popular emojis and potential hashtags.",
                "professional": "polished, direct, respectful, suitable for LinkedIn or portfolio sites.",
                "product": "clear features highlighted, brand-focused, encouraging purchasing/action.",
                "marketing": "persuasive, punchy, call-to-action oriented, highlighting benefits.",
                "casual": "relaxed, conversational, friendly tone."
            }
            tone = style_guide.get(style.lower(), style_guide["casual"])

            prompt = f"Write a single photo caption for this image in a {style.upper()} tone. Tone details: {tone} Respond ONLY with the caption text."

            if self.provider == "gemini":
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([prompt, img])
                caption_text = response.text or ""
                return self._clean_text(caption_text).strip().strip('"')
            else:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"

                prompt = f"Write a single photo caption for this image in a {style.upper()} tone. Tone details: {tone} Keep your thinking/reasoning process extremely brief (1-2 sentences), then output the caption text. Respond ONLY with the caption text."

                response = await self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                caption_text = response.choices[0].message.content or ""
                return self._clean_text(caption_text).strip().strip('"')
        
        try:
            return await self._retry_with_backoff(_caption_impl)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Caption generation failed after retries: {error_msg}")
            
            if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError("AI service authentication failed during caption generation. Please check your API key configuration.")
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("AI service rate limit exceeded during caption generation. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise RuntimeError("AI service timed out during caption generation. Please check your connection and try again.")
            elif "quota" in error_msg.lower():
                raise RuntimeError("AI service quota exceeded during caption generation. Please check your plan and usage.")
            else:
                raise RuntimeError(f"Unable to generate caption: {error_msg}. Please try again later.")

    async def generate_captions(self, image_bytes: bytes, style: str = "casual") -> list[str]:
        """Generate 3 unique captions with retry logic and enhanced error handling."""
        self._verify_configuration()
        
        async def _captions_impl():
            style_guide = {
                "instagram": "fun, engaging, casual with relevant popular emojis and potential hashtags.",
                "professional": "polished, direct, respectful, suitable for LinkedIn or portfolio sites.",
                "product": "clear features highlighted, brand-focused, encouraging purchasing/action.",
                "marketing": "persuasive, punchy, call-to-action oriented, highlighting benefits.",
                "casual": "relaxed, conversational, friendly tone."
            }
            tone = style_guide.get(style.lower(), style_guide["casual"])

            prompt = (
                f"Write exactly 3 different photo captions for this image in a {style.upper()} tone. "
                f"Tone details: {tone} "
                "Each caption should be unique and offer a different angle or wording. "
                "Respond ONLY with a valid JSON array of exactly 3 strings. "
                'Example: ["Caption one here.", "Caption two here.", "Caption three here."]'
            )

            if self.provider == "gemini":
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([prompt, img])
                raw_text = response.text or ""
                parsed = self._extract_json(raw_text)
                if isinstance(parsed, list):
                    return [str(c).strip().strip('"') for c in parsed[:3]]
                return [self._clean_text(raw_text).strip().strip('"')]
            else:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"
                response = await self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    temperature=0.8,
                    max_tokens=1024
                )
                raw_text = response.choices[0].message.content or ""
                raw_text = self._clean_text(raw_text)
                parsed = self._extract_json(raw_text)
                if isinstance(parsed, list):
                    return [str(c).strip().strip('"') for c in parsed[:3]]
                return [raw_text.strip().strip('"')]
        
        try:
            return await self._retry_with_backoff(_captions_impl)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Multiple captions generation failed after retries: {error_msg}")
            
            if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError("AI service authentication failed during captions generation. Please check your API key configuration.")
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("AI service rate limit exceeded during captions generation. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise RuntimeError("AI service timed out during captions generation. Please check your connection and try again.")
            elif "quota" in error_msg.lower():
                raise RuntimeError("AI service quota exceeded during captions generation. Please check your plan and usage.")
            else:
                raise RuntimeError(f"Unable to generate captions: {error_msg}. Please try again later.")

    async def suggest_backgrounds(self, image_bytes: bytes) -> list[str]:
        """Suggest backgrounds with retry logic and enhanced error handling."""
        self._verify_configuration()
        
        async def _suggestions_impl():
            system_prompt = (
                "You are a professional designer. Analyze the image and recommend 3 to 5 background placement ideas. "
                "Your recommendations should suggest solid colors, scenes, or textures that will make the subject pop. "
                "Respond ONLY with a valid JSON object containing the key 'suggestions' pointing to a list of strings.\n"
                'Example format: {"suggestions": ["Studio Soft Gray", "Sunlit Minimalist Office", "Vibrant Cyberpunk Streets"]}'
            )

            if self.provider == "gemini":
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes))
                response = await model.generate_content_async([system_prompt, img])
                raw_text = response.text or ""
                parsed = self._extract_json(raw_text)
            else:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"

                system_prompt = (
                    "You are a professional designer. Analyze the image and recommend 3 to 5 background placement ideas. "
                    "Your recommendations should suggest solid colors, scenes, or textures that will make the subject pop. "
                    "Keep your thinking/reasoning process extremely brief (1-2 sentences), then output the JSON object. "
                    "Respond ONLY with a valid JSON object containing the key 'suggestions' pointing to a list of strings.\n"
                    'Example format: {"suggestions": ["Studio Soft Gray", "Sunlit Minimalist Office", "Vibrant Cyberpunk Streets"]}'
                )

                response = await self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Suggest backgrounds for this image as a JSON object with key 'suggestions'."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1536
                )
                raw_text = response.choices[0].message.content or ""
                parsed = self._extract_json(raw_text)
            
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            elif isinstance(parsed, dict):
                for key in ["suggestions", "backgrounds", "ideas", "recommendations", "concepts"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return [str(item) for item in parsed[key]]
                for val in parsed.values():
                    if isinstance(val, list):
                        return [str(item) for item in val]
            
            raise ValueError(f"Could not extract background list from parsed JSON: {parsed}")
        
        try:
            return await self._retry_with_backoff(_suggestions_impl)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Background suggestions failed after retries: {error_msg}")
            
            if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError("AI service authentication failed during background suggestions. Please check your API key configuration.")
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("AI service rate limit exceeded during background suggestions. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise RuntimeError("AI service timed out during background suggestions. Please check your connection and try again.")
            elif "quota" in error_msg.lower():
                raise RuntimeError("AI service quota exceeded during background suggestions. Please check your plan and usage.")
            else:
                raise RuntimeError(f"Unable to generate background suggestions: {error_msg}. Please try again later.")
