import gradio as gr
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

groq = os.getenv("GROQ_API_KEY")

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

llm = ChatGroq(
    temperature=0.0,
    api_key=groq,
    model="llama-3.3-70b-versatile"
)

def classification_node(state: State):
    prompt = PromptTemplate(
        input_variable = ["text"],
        template = "Classify the following text into one of the following categories: News, Blog, Research or Other. DO NOT PROVIDE REASON.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content = prompt.format(text = state["text"]))

    classification  = llm.invoke([message]).content.strip()
    return {"classification":classification}

def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables = ["text"],
        template = "Extract all entities (Person, Organisation, Location, etc.) from the following text. Provide the result as a comma separated list. JUST RETURN THE LIST AND NOTHING ELSE.\n\n {text} \n\n Entities:"
    )

    message = HumanMessage(content = prompt.format(text = state["text"]))

    entities = llm.invoke([message]).content.strip().split(",  ")
    return {"entities":entities}

def summarization_node(state: State):
    prompt = PromptTemplate(
        input_variable = ["text"],
        template = "Summarize the following text into a small text. JUST RETURN THE SUMMARY AND NOTHING ELSE.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content = prompt.format(text = state["text"]))

    summary  = llm.invoke([message]).content.strip()
    return {"summary":summary}

workflow = StateGraph(State)

workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction_node", entity_extraction_node)
workflow.add_node("summarization_node", summarization_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction_node")
workflow.add_edge("entity_extraction_node", "summarization_node")
workflow.add_edge("summarization_node", END)

app = workflow.compile()

def func(text):
    state_input = {"text": text}
    result = app.invoke(state_input)
    return result["classification"], result["entities"], result["summary"]

demo = gr.Interface(fn=func,
                    inputs=gr.Textbox(placeholder="Enter text", label="Text"),
                    outputs=[gr.Textbox(placeholder="Classification", label="Classification"),gr.Textbox(placeholder="Entities", label="Entities"),gr.Textbox(placeholder="Summary", label="Summary")])

demo.launch()