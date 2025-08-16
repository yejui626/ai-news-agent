from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# -----------------------------
# 📘 Pydantic Schemas
# -----------------------------

class Company(BaseModel):
    """Extracted data about a company's news."""
    name: Optional[str] = Field(default=None, description="Name of the company")
    ticker: Optional[str] = Field(default=None, description="Ticker symbol of the company")
    content: List[str] = Field(default=None, description="News content related to the company")


class News(BaseModel):
    """Structured extraction of macroeconomic news and company-level news."""
    macro: List[str] = Field(default=None, description="Macroeconomic news")
    company: List[Company] = Field(default=None, description="Extracted data about companies' news.")


# -----------------------------
# 🤖 Summarizer Class
# -----------------------------

class NewsSummarizer:
    def __init__(self):
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value."
            ),
            ("human", "{text}")
        ])

        # LLM instance with structured output
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=0
        )

        self.structured_llm = self.llm.with_structured_output(schema=News)

    def summarize(self, text: str) -> dict:
        """
        Summarize and extract structured news content from raw text.
        :param text: Raw scraped content.
        :return: Dictionary with `macro` and `company` keys.
        """
        prompt = self.prompt_template.invoke({"text": text})
        output = self.structured_llm.invoke(prompt)
        return output.model_dump()
