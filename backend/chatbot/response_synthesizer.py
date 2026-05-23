import os
import json
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are an executive financial document analyst for DocParse.
Deliver concise, data-driven responses.

RULES:
- Max 3 sentences for simple answers
- Bullet points ONLY for lists of 4 or more items
- Always include key numbers with currency symbol
- Never say "Would you like to..." or "I hope this helps"
- Never repeat the question back
- If no data: say "No results found." and suggest one reason
- For comparisons: state the winner first, then the difference
- Lead with the most important number or insight
- If question starts with "show", "list", "display", "give me all" → 
  return a structured list of results, not a paragraph summary
- For list queries with more than 5 results → show first 5 with format:
  1. filename | vendor | amount | currency
  2. ...
  and end with "... and X more results"
- Never summarize a list query into a paragraph

CONVERSATION HISTORY:
{history}

QUESTION: {original_question}
ACTUAL QUERY: {rewritten_question}
RESULTS ({count} rows): {results}

Response:
""")

GREETING_PROMPT = ChatPromptTemplate.from_template("""
You are a friendly assistant for DocParse — an AI document processing system.
Respond naturally to the greeting or general question.
Keep it brief, warm, and helpful.
Mention 2-3 things the user can ask about (invoices, contracts, analytics).

Conversation history:
{history}

Question: {question}

Response:
""")

CLARIFICATION_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful document analytics assistant.
The user wants clarification about your previous answer.
Explain it clearly and simply.

CONVERSATION HISTORY:
{history}

CLARIFICATION REQUEST:
{question}

Response:
""")


def synthesize_response(
    original_question: str,
    rewritten_question: str,
    results: list,
    history: str,
    intent: str
) -> str:
    """Synthesize a natural language response from query results."""

    try:
        # Handle greetings
        if intent == "greeting":
            chain = GREETING_PROMPT | llm
            response = chain.invoke({
                "question": original_question,
                "history": history or "No history"
            })
            return response.content.strip()

        # Handle clarification
        if intent == "clarification":
            chain = CLARIFICATION_PROMPT | llm
            response = chain.invoke({
                "question": original_question,
                "history": history or "No history"
            })
            return response.content.strip()

        # Format results for prompt
        results_str = json.dumps(results[:30], indent=2) if results else "[]"

        chain = SYNTHESIS_PROMPT | llm
        response = chain.invoke({
            "original_question": original_question,
            "rewritten_question": rewritten_question,
            "results": results_str,
            "count": len(results),
            "history": history or "No history"
        })

        return response.content.strip()

    except Exception as e:
        logger.error(f"Response synthesis failed: {e}")
        if results:
            return f"Found {len(results)} results. Here are the first few: {json.dumps(results[:3], indent=2)}"
        return "I could not format the response. Please try again."


def synthesize_error(question: str, error: str) -> str:
    """Generate a helpful error message."""
    return f"I could not process that query. Try rephrasing — for example: 'Show all invoices' or 'Top vendors by amount'."


def synthesize_general(question: str, history: str) -> str:
    """Handle general questions."""
    try:
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for DocParse document processing system.
Answer clearly and concisely. Stay focused on document processing topics.

Conversation history:
{history}

Question: {question}
Answer:
""")
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "history": history or "No history"
        })
        return response.content.strip()
    except Exception as e:
        logger.error(f"General synthesis failed: {e}")
        return "I can help you analyze your documents. Try asking about invoices, contracts, or vendors."

def synthesize_multi(
    original_question: str,
    sub_results: list[dict],
    history: str
) -> str:
    """Synthesize response from multiple sub-question results."""
    try:
        prompt = ChatPromptTemplate.from_template("""
You are an executive financial analyst.
Combine these multiple query results into one concise analytical response.

RULES:
- Lead with the most important insight
- Max 5 sentences total
- Include all key numbers with currency
- Never say "Would you like..." or "I hope this helps"
- Be direct and analytical

ORIGINAL QUESTION: {question}
SUB-RESULTS: {results}
HISTORY: {history}

Response:
""")
        chain = prompt | llm
        response = chain.invoke({
            "question": original_question,
            "results": str(sub_results)[:2000],
            "history": history or "No history"
        })
        return response.content.strip()
    except Exception as e:
        logger.error(f"Multi synthesis failed: {e}")
        return "\n\n".join([r.get("answer", "") for r in sub_results])