from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state of our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the node in our graph
def chatbot(state: AgentState) -> AgentState:
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Get the messages from the state
    messages = state["messages"]
    
    # Generate a response
    response = llm.invoke(messages)
    
    # Add the AI's response to the messages
    return {"messages": [*messages, response]}


# Build the graph
def build_graph():
    # Initialize the graph
    builder = StateGraph(AgentState)
    
    builder.add_node("chatbot", chatbot)

    builder.set_entry_point("chatbot")
    
    builder.add_edge("chatbot", END)

    # Compile the graph
    return builder.compile()


# Create a function to run the graph
def main():
    # Build the graph
    graph = build_graph()
    
    # Initialize the state
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful AI assistant."),
        ]
    }
    
    # Start with the initial state
    state = initial_state
    
    print("Chat with the AI assistant. Type '/bye' to exit.\n")
    
    # Run in an infinite loop until the user types '/bye'
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if the user wants to exit
        if user_input.lower() == "/bye":
            print("\nAI: Goodbye! Have a great day!")
            break
        
        # Add the user message to the state
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph with the updated state
        state = graph.invoke(state)
        
        # Get the latest message and print it
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage):
            print(f"\nAI: {messages[-1].content}\n")


# Run the application
if __name__ == "__main__":
    main()
