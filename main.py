
# main.py
import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

# import tools
from tools import add, multiply

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")

# Initialize Chat Model
llm = ChatOpenAI(model = "gpt-4o-mini")

# Bind Tools
llm_with_tools = llm.bind_tools(tools=[add,multiply])

# Query the LLM
query = "What is 7 * 8 and 11 + 15"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)
if response.invalid_tool_calls:
    print("Invalid tool call detected!")
messages.append(response)

for tool_call in response.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id = tool_call["id"]))

# Print Results
answer = llm_with_tools.invoke(messages)
answer.pretty_print()

