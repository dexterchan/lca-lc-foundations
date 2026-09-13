# %% [markdown]
# ## Tool definition

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# %%
@tool("square_root")
def tool1(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

# %%
@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5

# %%
tool1.invoke({"x": 467})

# %% [markdown]
# ## Adding to agents

# %%
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model(
    model="deepseek/deepseek-v4-flash-0731",
    model_provider="deepseek",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

agent = create_agent(
    model=model,
    tools=[tool1],
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number."
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content="What is the square root of 467?")

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)

# %%
from pprint import pprint

pprint(response['messages'])

# %%
print([a.tool_calls for a in response["messages"] if getattr(a, 'tool_calls', None)])
# why getattr(response["messages"][0], 'tool_calls') not working?

# %% [markdown]
# ## Handling tool errors
#
# First attempt below: let the tool raise and hope the agent catches it. Running it
# shows that's wrong for this LangChain version — the exception blows straight through
# and kills the process. So the real rule is: **never let a tool function raise.**
# Catch inside the tool, always, and return a plain-text explanation instead. That's
# the same contract every well-behaved MCP tool follows (Outlook, file I/O, auth
# checks, etc.) — a failure is just a message, never a crash.

# %%
@tool
def safe_divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

error_agent = create_agent(
    model=model,
    tools=[safe_divide],
    system_prompt="You are an arithmetic wizard. Use your tools to do division."
)

response = error_agent.invoke(
    {"messages": [HumanMessage(content="What is 10 divided by 0?")]}
)

# Look for the ToolMessage in here — LangChain caught the ZeroDivisionError for you
# and reported it back to the model as a failed tool call, no crash.
pprint(response["messages"])
print(response["messages"][-1].content)

# %%
@tool
def safe_divide_handled(a: float, b: float) -> str:
    """Divide a by b. Returns a friendly error message instead of raising."""
    try:
        return str(a / b)
    except ZeroDivisionError:
        return "Error: cannot divide by zero. Ask the user for a non-zero denominator."

handled_agent = create_agent(
    model=model,
    tools=[safe_divide_handled],
    system_prompt="You are an arithmetic wizard. Use your tools to do division."
)

response = handled_agent.invoke(
    {"messages": [HumanMessage(content="What is 10 divided by 0?")]}
)

print(response["messages"][-1].content)

# %% [markdown]
# ## A reusable pattern for every tool
#
# Writing `try/except` by hand inside every tool is easy to forget. Wrap it once in a
# decorator instead — any tool you build gets soft-fail behavior automatically, no
# matter what kind of error it throws (network timeout, bad auth, division by zero,
# whatever). This is the same shape MCP servers use at their tool boundary.

# %%
import functools

def soft_fail(fn):
    """Wrap a tool's plain function so any exception becomes a friendly message
    instead of crashing the agent. Put this closest to the raw function, below @tool."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # The model reads this text and can decide what to do next
            # (retry with different input, apologize, ask the user for help).
            return f"Tool '{fn.__name__}' failed: {e}"
    return wrapper

# %%
@tool
@soft_fail
def safe_divide_v2(a: float, b: float) -> str:
    """Divide a by b."""
    return str(a / b)

@tool
@soft_fail
def square_root_v2(x: float) -> str:
    """Calculate the square root of a number. Only works for x >= 0."""
    if x < 0:
        raise ValueError("square_root only accepts non-negative numbers")
    return str(x ** 0.5)

robust_agent = create_agent(
    model=model,
    tools=[safe_divide_v2, square_root_v2],
    system_prompt="You are an arithmetic wizard. Use your tools for division and square roots."
)

response = robust_agent.invoke(
    {"messages": [HumanMessage(content="What is 10 divided by 0? Also, what is the square root of -9?")]}
)

print(response["messages"][-1].content)


response = robust_agent.invoke(
    {"messages": [HumanMessage(content="What is square root of 1000?")]}
)

print(response["messages"][-1].content)
