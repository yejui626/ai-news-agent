from langgraph.graph import StateGraph, START, MessagesState, END
from .agents import parse_headlines_agent, parse_blog_content_agent
from .handoff import assign_to_parse_headlines_agent, assign_to_parse_blog_content_agent
from .utils import pretty_print_messages


supervisor_agent = None


from langgraph.prebuilt import create_react_agent
from .agents import llm

supervisor_agent = create_react_agent(
    model=llm,
    tools=[assign_to_parse_headlines_agent, assign_to_parse_blog_content_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a news headlines agent. Assign headlines research (including title, author, urls, date) tasks to this agent\n"
        "- a blog content agent. Assign extracting detailed blog content tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "After all agents have completed their tasks, answer the user's question using the collected outputs."
    ),
    name="supervisor",
)


# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(supervisor_agent, destinations=("parse_headlines_agent", "parse_blog_content_agent", END))
    .add_node(parse_headlines_agent)
    .add_node(parse_blog_content_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("parse_headlines_agent", "supervisor")
    .add_edge("parse_blog_content_agent", "supervisor")
    .compile()
)


def run_supervisor(user_question: str = "What is the news headlines for today?"):
    """Run the supervisor graph and stream printed updates."""
    final_message_history = None
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question,
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)
        final_message_history = chunk.get("supervisor", {}).get("messages")

    return final_message_history