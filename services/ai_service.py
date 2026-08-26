import os
import base64
import json
import re
import asyncio
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict, Tuple
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Guard optional AI-provider packages
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
    """Handles communication with AI providers (Gemini or Groq) with enhanced error handling and retry logic."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.is_configured: bool = False
        self.provider: str = "gemini"
        self.api_key: str = ""
        self.chat_model: str = "gemini-3.6-flash"
        self.vision_model: str = "gemini-3.6-flash"
        self.client: Optional[Any] = None
        self._setup_provider()

    def _setup_provider(self) -> None:
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        if self.provider == "groq":
            if not _openai_available or AsyncOpenAI is None:
                raise ImportError(
                    "The 'openai' package is not installed for Groq support. Run: pip install openai"
                )
            self.api_key = os.getenv("GROQ_API_KEY", "")
            self.is_configured = bool(self.api_key and "your_groq_api_key" not in self.api_key)
            self.chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
            self.vision_model = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
            if self.is_configured:
                self.client = AsyncOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key,
                )
            else:
                self.client = None
        else:
            # Default to Gemini
            self.provider = "gemini"
            if not _genai_available or genai is None:
                raise ImportError(
                    "The 'google-generativeai' package is not installed. Run: pip install google-generativeai"
                )
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            self.is_configured = bool(self.api_key and "your_gemini_api_key" not in self.api_key)
            if self.is_configured:
                genai.configure(api_key=self.api_key)
            self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
            self.vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")
            self.client = None

    async def _retry_with_backoff(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Retry an async function with exponential backoff."""
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                # Don't retry on certain non-transient errors
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

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Function {func.__name__} failed without raising an exception.")

    def _verify_configuration(self) -> None:
        """Verify AI configuration with helpful error messages."""
        if not self.is_configured:
            # Try to reload environment variables
            env_path = Path(__file__).resolve().parent.parent / ".env"
            load_dotenv(dotenv_path=env_path, override=True)
            self._setup_provider()

        # Final check with user-friendly error message
        if not self.is_configured:
            provider_name = "Groq" if self.provider == "groq" else "Gemini"
            key_name = "GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY"
            placeholder = "your_groq_api_key_here" if self.provider == "groq" else "your_gemini_api_key_here"

            # Check if placeholder key is still being used
            if self.api_key and ("your_" in self.api_key.lower() or "placeholder" in self.api_key.lower()):
                raise ValueError(
                    f"{provider_name} API key is still set to a placeholder value. "
                    f"Please replace '{placeholder}' with your actual API key in the .env file."
                )

            raise ValueError(
                f"{provider_name} API key is not configured or invalid. "
                f"Please set a valid {key_name} in the .env file. "
                f"Make sure the .env file exists in the backend directory and contains your API key."
            )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        return text.strip()

    def _extract_json(self, text: str) -> Any:
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

    def _extract_thinking(self, text: str) -> Tuple[str, Optional[str]]:
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
        else:
            open_match = re.search(r'<think>(.*)', text, re.DOTALL)
            thinking = open_match.group(1).strip() if open_match else None
        reply = self._clean_text(text)
        return reply, thinking

    async def chat(self, message: str, image_bytes: Optional[bytes] = None) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """Chat with AI assistant with retry logic, action extraction, and enhanced error handling."""
        self._verify_configuration()

        system_instructions = (
            "You are a professional designer and helpful assistant for the AI Background Remover application. "
            "You are answering user queries about the uploaded image. Help them with background recommendations, editing advice, and captions.\n"
            "If the user asks you to perform an editing action (such as applying a white background, solid color background, a studio background, adjusting enhancement settings like brightness/contrast/saturation, or cropping the image to a specific aspect ratio), "
            "you must append a single line to the very end of your response starting with '[ACTION:' and ending with ']' containing a JSON payload detailing the requested action. "
            "Do not include code block ticks in the action line. Keep it as a raw single line.\n"
            "Examples:\n"
            "- To change background color: [ACTION: {\"type\": \"apply_bg\", \"bgType\": \"solid\", \"solidColor\": \"#ffffff\"}]\n"
            "- To apply a library background: [ACTION: {\"type\": \"apply_bg\", \"bgType\": \"library\", \"libraryUrl\": \"https://images.unsplash.com/photo-1553356084-58ef4a67b2a7?w=1200\"}]\n"
            "- To enhance settings: [ACTION: {\"type\": \"apply_enhance\", \"brightness\": 1.2, \"contrast\": 1.1, \"saturation\": 1.0, \"sharpness\": 1.2, \"denoise\": false, \"auto_wb\": true, \"denoise_strength\": 9}]\n"
            "- To crop settings: [ACTION: {\"type\": \"apply_crop\", \"aspectRatio\": \"1:1\", \"paddingPct\": 0.1}]\n"
            "Make sure to return only valid action payloads when requested."
        )

        async def _chat_impl() -> Tuple[str, Optional[str]]:
            if self.provider == "gemini":
                if genai is None:
                    raise RuntimeError("Gemini package is not available.")
                model = genai.GenerativeModel(self.chat_model)
                if image_bytes:
                    img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    prompt = (
                        f"{system_instructions}\n\n"
                        f"User Message: {message}"
                    )
                    response = await model.generate_content_async([prompt, img])
                else:
                    prompt = (
                        f"{system_instructions}\n\n"
                        f"User Message: {message}"
                    )
                    response = await model.generate_content_async(prompt)

                raw_reply = response.text or "No response from AI."
                return self._extract_thinking(raw_reply)
            else:
                if self.client is None:
                    raise RuntimeError("Groq client is not initialized.")
                messages: List[Dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": system_instructions
                    }
                ]
                if image_bytes:
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    })
                else:
                    messages.append({"role": "user", "content": message})

                response = await self.client.chat.completions.create(
                    model=self.vision_model if image_bytes else self.chat_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                raw_reply = response.choices[0].message.content or "No response from AI."
                return self._extract_thinking(raw_reply)

        try:
            reply, thinking = await self._retry_with_backoff(_chat_impl)
            
            # Extract and parse action block if present
            action = None
            action_match = re.search(r'\[ACTION:\s*(\{.*?\})\s*\]', reply)
            if action_match:
                try:
                    action = json.loads(action_match.group(1).strip())
                    reply = re.sub(r'\[ACTION:\s*\{.*?\}\s*\]', '', reply).strip()
                except Exception as e:
                    logger.warning(f"Failed to parse action JSON: {e}")
            
            return reply, thinking, action
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chat failed after retries: {error_msg}")

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

    async def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze image with retry logic and enhanced error handling."""
        self._verify_configuration()

        async def _analyze_impl() -> Dict[str, Any]:
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
                if genai is None:
                    raise RuntimeError("Gemini package is not available.")
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([system_prompt, img])
                raw_text = response.text or ""
                return self._extract_json(raw_text)
            else:
                if self.client is None:
                    raise RuntimeError("Groq client is not initialized.")
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"

                prompt = (
                    f"{system_prompt}\n"
                    "Keep your thinking/reasoning process extremely brief (1-2 sentences), then output the JSON object. "
                    "Respond ONLY with the JSON object."
                )

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
                    max_tokens=1536
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

        async def _caption_impl() -> str:
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
                if genai is None:
                    raise RuntimeError("Gemini package is not available.")
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([prompt, img])
                caption_text = response.text or ""
                return self._clean_text(caption_text).strip().strip('"')
            else:
                if self.client is None:
                    raise RuntimeError("Groq client is not initialized.")
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

    async def generate_captions(self, image_bytes: bytes, style: str = "casual") -> List[str]:
        """Generate 3 unique captions with retry logic and enhanced error handling."""
        self._verify_configuration()

        async def _captions_impl() -> List[str]:
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
                if genai is None:
                    raise RuntimeError("Gemini package is not available.")
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([prompt, img])
                raw_text = response.text or ""
                parsed = self._extract_json(raw_text)
                if isinstance(parsed, list):
                    return [str(c).strip().strip('"') for c in parsed[:3]]
                return [self._clean_text(raw_text).strip().strip('"')]
            else:
                if self.client is None:
                    raise RuntimeError("Groq client is not initialized.")
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

    async def suggest_backgrounds(self, image_bytes: bytes) -> List[str]:
        """Suggest backgrounds with retry logic and enhanced error handling."""
        self._verify_configuration()

        async def _suggestions_impl() -> List[str]:
            system_prompt = (
                "You are an expert art director and visual stylist for a professional photography and design studio. "
                "Carefully inspect the uploaded image to identify the exact subject (e.g. animal/bird/pet, product/handbag/shoe, beauty/cosmetic, portrait, food, electronics), "
                "its dominant color palette, lighting temperature, and aesthetic mood.\n\n"
                "Generate exactly 4 distinct, highly relevant background recommendations that perfectly match the vibe, context, and color harmony of this specific subject:\n"
                "1. Natural/Environmental Context Backdrop (a realistic, picturesque setting where the subject naturally thrives or looks stunning)\n"
                "2. Professional Studio / Aesthetic Texture Backdrop (a clean, luxurious studio texture like marble, terracotta, linen, dark concrete, or warm wood that makes the subject pop)\n"
                "3. Harmonious Color / Gradient Theme (a complementary or contrasting color theme based on the subject's palette, e.g. 'Deep Indigo Matte Studio' or 'Warm Terracotta Wall')\n"
                "4. Creative / Atmospheric Scene (an atmospheric backdrop like sunlit botanical garden, golden hour meadow, blurred cinematic bokeh, or modern architecture)\n\n"
                "Ensure every suggestion is descriptive and evocative (e.g., 'Misty Amazonian Rainforest Canopy', 'Warm Terracotta Clay Wall', 'Minimalist Italian White Marble Studio', 'Sunlit Blooming Hibiscus Garden'). "
                "Respond ONLY with a valid JSON object containing the key 'suggestions' pointing to an array of 4 descriptive strings.\n"
                'Example: {"suggestions": ["Misty Amazonian Rainforest Canopy", "Warm Terracotta Clay Wall", "Sunlit Blooming Hibiscus Garden", "Deep Indigo Matte Studio"]}'
            )

            if self.provider == "gemini":
                if genai is None:
                    raise RuntimeError("Gemini package is not available.")
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes))
                response = await model.generate_content_async([system_prompt, img])
                raw_text = response.text or ""
                parsed = self._extract_json(raw_text)
            else:
                if self.client is None:
                    raise RuntimeError("Groq client is not initialized.")
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"

                prompt = (
                    "You are a professional designer. Analyze the image and recommend 3 to 5 background placement ideas. "
                    "Your recommendations should suggest solid colors, scenes, or textures that will make the subject pop. "
                    "Keep your thinking/reasoning process extremely brief (1-2 sentences), then output the JSON object. "
                    "Respond ONLY with a valid JSON object containing the key 'suggestions' pointing to a list of strings.\n"
                    'Example format: {"suggestions": ["Studio Soft Gray", "Sunlit Minimalist Office", "Vibrant Cyberpunk Streets"]}'
                )

                response = await self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {"role": "system", "content": prompt},
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

    async def analyze_image_advanced(self, image_bytes: bytes) -> Dict[str, Any]:
        """Perform advanced analysis including object detection, color extraction, style recommendations, and composition analysis."""
        self._verify_configuration()

        async def _analyze_impl() -> Dict[str, Any]:
            system_prompt = (
                "You are an expert AI design assistant specializing in image analysis, object detection, color theory, and composition.\n"
                "Analyze the image and return a JSON object matching this schema:\n"
                "{\n"
                "  \"object_detection\": [\n"
                "    {\n"
                "      \"label\": \"Name of the detected object (e.g. Person, Handbag, Wine Bottle, Cat)\",\n"
                "      \"box_2d\": [ymin, xmin, ymax, xmax],\n"\
                "      \"confidence\": 0.95\n"
                "    }\n"
                "  ],\n"
                "  \"color_palette\": [\n"
                "    {\n"
                "      \"hex\": \"#HEXCODE (e.g., #3A5F43)\",\n"
                "      \"name\": \"Friendly color name (e.g., Forest Green)\",\n"
                "      \"percentage\": 40,\n"
                "      \"text_color\": \"#ffffff or #000000 (suitable for text overlay on this color)\",\n"
                "      \"use_case\": \"Design advice for this color (e.g., Use as a background accent to highlight warm subject elements.)\"\n"
                "    }\n"
                "  ],\n"
                "  \"style_transfer\": [\n"
                "    {\n"
                "      \"style\": \"Name of recommended filter/artistic style (e.g. Vaporwave, Cyberpunk, Cinematic Studio, Minimalist Pastel, Pop Art)\",\n"
                "      \"description\": \"Description of the style recommendations for the subject.\",\n"
                "      \"prompts\": \"Suggested text prompts to generate backgrounds in this style\"\n"
                "    }\n"
                "  ],\n"
                "  \"composition\": {\n"
                "    \"rule_of_thirds\": \"Analysis of how subject alignment adheres to or breaks the rule of thirds.\",\n"
                "    \"leading_lines\": \"Analysis of leading lines in the image and how they guide the viewer's eye.\",\n"
                "    \"balance\": \"Assessment of visual balance (e.g. Symmetric, Asymmetric, Radial) and weight distribution.\",\n"
                "    \"crop_recommendation\": \"Recommendations for cropping or positioning to enhance composition.\"\n"
                "  },\n"
                "  \"suggested_backgrounds\": [\n"
                "    \"Suggest a solid color hex code (e.g., #FFFFFF) or library backdrop term (e.g., Soft White Studio, Light Grey Wall, Marble Surface, Green Forest, Autumn Leaves, City Skyline, Night Streets, Modern Office)\"\n"
                "  ],\n"
                "  \"optimal_enhancement\": {\n"
                "    \"brightness\": 1.0,\n"
                "    \"contrast\": 1.0,\n"
                "    \"saturation\": 1.0,\n"
                "    \"sharpness\": 1.0,\n"
                "    \"denoise\": false,\n"
                "    \"auto_wb\": false,\n"
                "    \"denoise_strength\": 9\n"
                "  },\n"
                "  \"suggested_crop\": {\n"
                "    \"aspect_ratio\": \"free\",\n"
                "    \"padding_pct\": 0.05\n"
                "  },\n"
                "  \"suggested_filename\": \"Descriptive, SEO-friendly filename based on the subject (e.g., warm_knit_sweater_portrait or hydrate_serum_cosmetic_bottle). Format as lowercase snake_case without extension.\"\n"
                "}\n\n"
                "Ensure all box_2d coordinate values are integers in the range [0, 100] representing percentages of the image height and width.\n"
                "Ensure suggested_crop.aspect_ratio is one of: 'free', '1:1', '4:3', '3:4', '16:9', '9:16', '3:2', '2:3', '5:4', '4:5'.\n"
                "Respond ONLY with a valid JSON object matching this structure."
            )

            if self.provider == "gemini":
                if genai is None:
                    raise RuntimeError("Gemini package is not available.")
                model = genai.GenerativeModel(self.vision_model)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                response = await model.generate_content_async([system_prompt, img])
                raw_text = response.text or ""
                return self._extract_json(raw_text)
            else:
                if self.client is None:
                    raise RuntimeError("Groq client is not initialized.")
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{base64_image}"

                prompt = (
                    f"{system_prompt}\n"
                    "Keep your thinking/reasoning process extremely brief (1-2 sentences), then output the JSON object. "
                    "Respond ONLY with the JSON object."
                )

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
                    max_tokens=2048
                )
                raw_text = response.choices[0].message.content or ""
                return self._extract_json(raw_text)

        try:
            return await self._retry_with_backoff(_analyze_impl)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Advanced image analysis failed after retries: {error_msg}")

            if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError("AI service authentication failed during advanced analysis. Please check your API key configuration.")
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("AI service rate limit exceeded during advanced analysis. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise RuntimeError("AI service timed out during advanced analysis. Please check your connection and try again.")
            elif "quota" in error_msg.lower():
                raise RuntimeError("AI service quota exceeded during advanced analysis. Please check your plan and usage.")
            else:
                raise RuntimeError(f"Unable to perform advanced analysis: {error_msg}. Please try again later.")
