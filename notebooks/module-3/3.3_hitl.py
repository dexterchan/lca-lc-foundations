# %%
from dotenv import load_dotenv
import langchain
load_dotenv()

# %%
from langchain.tools import tool, ToolRuntime


@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    # Take email from state.
    return runtime.state["email"]


@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    # Fake email sending.
    return "Email sent"


# %%
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware


class EmailState(AgentState):
    email: str


model = init_chat_model(
    model="deepseek/deepseek-v4-flash-0731",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

agent = create_agent(
    model=model,
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)

# %%
from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}
email_request = {
    "messages": [
        HumanMessage(
            content="Please read my email and send a response immediately. Send the reply now in the same thread."
        )
    ],
    "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John.",
}

response = agent.invoke(email_request, config=config)

# %%
from pprint import pprint

pprint(response)

# %%
print(response["__interrupt__"])

# %%
# Access just the 'body' argument from the tool call.
print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])
pprint([m for m in response['messages'] if isinstance(m, langchain.messages.AIMessage)])
# %% [markdown]
# ## Approve

# %%
from langgraph.types import Command

response = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,  # Same thread ID to resume the paused conversation.
)

pprint(response)

# %% [markdown]
# ## Reject

# %%
# Start a new thread: the previous send was already approved.
config = {"configurable": {"thread_id": "2"}}
response = agent.invoke(email_request, config=config)

response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    # An explanation of why the request was rejected.
                    "message": "No please sign off - Your merciful leader, Seán.",
                }
            ]
        }
    ),
    config=config,  # Same thread ID to resume the paused conversation.
)

pprint(response)

# %%
# If the agent tries sending a revised reply, it pauses for approval again.
if response.get("__interrupt__"):
    print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])

# %% [markdown]
# ## Edit

# %%
# Give the edit example its own pending send.
config = {"configurable": {"thread_id": "3"}}
response = agent.invoke(email_request, config=config)

response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "send_email",
                        "args": {"body": "This is the last straw, you're fired!"},
                    },
                }
            ]
        }
    ),
    config=config,  # Same thread ID to resume the paused conversation.
)

pprint(response)

# %%
