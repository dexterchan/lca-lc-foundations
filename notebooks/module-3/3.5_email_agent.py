# %% [markdown]
# ## Email agent
# Sign in locally, check the sample inbox, draft a reply, and approve the send.
# The inbox and email sending tools use fake data.
# Set EMAIL_AGENT_DEMO_PASSWORD in your local .env file before running.
# Enter that password only at the hidden local prompt, never in chat.

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
import asyncio
import getpass
import hmac
import os
from dataclasses import dataclass
from typing import Awaitable, Callable
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.agents import AgentState


# %% [markdown]
# ## Local sign-in
# This function is ordinary Python, not an LLM tool or a graph node.
# Passwords never enter graph inputs, state, context, or tool arguments.

# %%
@dataclass(frozen=True)
class EmailSession:
    """An account verified by trusted local application code."""

    email_address: str

class CustomState(AgentState):
    email_session: EmailSession

def authenticate_locally(email: str, password: str) -> EmailSession:
    """Check the demo login locally. Do not expose this function as an AI tool."""
    expected_email = os.environ.get("EMAIL_AGENT_DEMO_EMAIL", "julie@example.com")
    expected_password = os.environ.get("EMAIL_AGENT_DEMO_PASSWORD")
    if not expected_password:
        raise RuntimeError("Set EMAIL_AGENT_DEMO_PASSWORD in your local .env file.")

    email_matches = hmac.compare_digest(email.encode(), expected_email.encode())
    password_matches = hmac.compare_digest(password.encode(), expected_password.encode())
    if not (email_matches and password_matches):
        raise PermissionError("Sign-in failed.")

    return EmailSession(email_address=email)


# %%
model = init_chat_model(
    model="deepseek/deepseek-v4-flash-0731",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

SIGN_IN_MESSAGE = "Please sign in locally before using email. Do not put passwords in chat."


def build_agent(*, session: EmailSession | None = None, checkpointer=None):
    """Bind an agent to a locally verified session, outside model-controlled state.

    Only trusted application code may supply a session after authentication.
    A server must not build it from client-supplied state or context.
    """
    if session is not None and not isinstance(session, EmailSession):
        raise TypeError("A locally verified EmailSession is required.")

    def require_session(email_session: EmailSession | None = None):
        if session is None:
            raise PermissionError("Sign in locally before using email.")
        if email_session != session:
            raise PermissionError("Email session in state does not match the signed-in session.")

    @tool
    def check_inbox(runtime: ToolRuntime) -> str:
        """Check the sample inbox for recent emails."""
        require_session(runtime.state.get("email_session"))
        return (
            "Hi Julie, I'm going to be in town next week and was wondering "
            "if we could grab a coffee? "
            "- best, Jane (jane@example.com)"
        )

    @tool
    def send_email(to: str, subject: str, body: str, runtime: ToolRuntime) -> str:
        """Send a response email (simulated for this lesson)."""
        require_session(runtime.state.get("email_session"))
        return f"Email sent to {to} with subject {subject} and body {body}"


    @wrap_model_call
    async def dynamic_tool_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        if session is None:
            # Fail locally: neither chat text nor a state flag can grant access.
            return AIMessage(content=SIGN_IN_MESSAGE)
        return await handler(request.override(tools=[check_inbox, send_email]))

    @dynamic_prompt
    def dynamic_prompt_func(request: ModelRequest) -> str:
        if session is None:
            return SIGN_IN_MESSAGE
        return (
            "You are a helpful assistant that can check the inbox and send emails. "
            "The user has already signed in through the application. "
            "Never ask for login credentials. Check the inbox before drafting a reply."
        )

    return create_agent(
        model=model,
        tools=[check_inbox, send_email],
        checkpointer=checkpointer,
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            HumanInTheLoopMiddleware(
                interrupt_on={"check_inbox": False, "send_email": True},
            ),
        ],
        state_schema=CustomState
    )


# Studio supplies persistence. This exported graph stays signed out.
# A real host app must authenticate first, then bind its own verified session.
agent = build_agent()

# %% [markdown]
# ## Run the full demo
# Login happens before the graph runs. Only email tasks go into its messages.

# %%
async def run_demo():
    """Sign in locally, then draft and approve a simulated email."""
    if not os.environ.get("EMAIL_AGENT_DEMO_PASSWORD"):
        raise RuntimeError("Set EMAIL_AGENT_DEMO_PASSWORD in your local .env file.")
    email = os.getenv("EMAIL_AGENT_DEMO_EMAIL") #input("Email [julie@example.com]: ").strip() or "julie@example.com"
    password = os.getenv("EMAIL_AGENT_DEMO_PASSWORD") #getpass.getpass("Local demo password: ")
    try:
        session = authenticate_locally(email, password)
    finally:
        del password

    demo_agent = build_agent(session=session, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    def new_run_id(run_name: str) -> dict:
        """Give this turn its own run_id so it shows up as its own trace in LangSmith."""
        run_id = str(uuid4())
        print(f"LangSmith run_id ({run_name}): {run_id}")
        return {**config, "run_id": run_id, "run_name": run_name}

    response = await demo_agent.ainvoke(
        {
            "messages": [HumanMessage(
                content="Check my inbox. Do not send a reply yet."
            )],
            "email_session": session,
        },
        config=new_run_id("check_inbox_turn"),
    )
    print(response["messages"][-1].content)

    response = await demo_agent.ainvoke(
        {"messages": [HumanMessage(
            content="Draft and send one reply to Jane accepting her coffee invitation. Any draft is fine."
        )]},
        config=new_run_id("draft_reply_turn"),
    )
    interrupts = response.get("__interrupt__")
    if not interrupts:
        raise RuntimeError(
            "The agent did not request approval to send an email. "
            "Its last reply was: " + str(response["messages"][-1].content)
        )

    action_requests = interrupts[0].value["action_requests"]
    for action in action_requests:
        print("\nDraft awaiting approval:")
        print("To:", action["args"]["to"])
        print("Subject:", action["args"]["subject"])
        print(action["args"]["body"])

    # This lesson approves a simulated send, with no actual email service.
    response = await demo_agent.ainvoke(
        Command(resume={
            "decisions": [{"type": "approve"} for _ in action_requests]
        }),
        config=new_run_id("approve_send_turn"),
    )
    print("\nAfter approval:")
    print(response["messages"][-1].content)
    return response


# %%
# In Jupyter, run this in a new cell after loading the definitions:
# result = await run_demo()
if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        result = asyncio.run(run_demo())
    else:
        print("In Jupyter, run this in a new cell: result = await run_demo()")
