import os
import logging
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


logger = logging.getLogger(__name__)

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )

def get_fallback_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("FALLBACK_LLM_MODEL", "gemini-3.1-flash-lite"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

def _extract_text(content) -> str:
    """
    Gemini (with AFC enabled) sometimes returns content as a list of
    content blocks instead of a plain string. Normalize to plain text.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "".join(parts).strip()
    return str(content).strip()
    
def invoke_with_fallback(chain_builder, inputs: dict) -> str:
    for llm_instance in [get_llm(), get_fallback_llm()]:
        try:
            chain = chain_builder(llm_instance)
            result = chain.invoke(inputs)
            return _extract_text(result.content)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                logger.warning(f"Rate limit hit on {llm_instance.__class__.__name__}, trying fallback")
                continue
            raise
    return ""

