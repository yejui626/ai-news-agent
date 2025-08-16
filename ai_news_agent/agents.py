from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from .tools import parse_headlines_agent as headlines_tool
from .tools import parse_blog_content_agent as blog_tool


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    # reasoning_format="hidden",
    timeout=None,
    max_retries=0,
    # other params...
)


parse_headlines_agent = create_react_agent(
    model=llm,
    tools=[headlines_tool],
    prompt=(
        "You are a headlines research (including title, author, urls, date) agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY getting the headlines tasks, DO NOT do any deep crawling\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="parse_headlines_agent",
)


parse_blog_content_agent = create_react_agent(
    model=llm,
    tools=[blog_tool],
    prompt=(
        "You are a web-scraping for detailed blog page agent. DO NOT use this if user only request to know the headlines\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with extracting blog post content from url and extracts clean text and images tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="parse_blog_content_agent",
)