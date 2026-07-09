


"""
retrieval.py

Loads the Chroma vector database, retrieves relevant documents,
formats the context, and sends the same retrieved context to
multiple LLMs for comparison.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel




###############################################################
# Load Environment Variables
###############################################################

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    print("WARNING : OPENAI_API_KEY not found in .env")
else:
    print("OpenAI API Key Loaded")


###############################################################
# Configuration
###############################################################

CHROMA_DB_DIR = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


###############################################################
# Embedding Model
###############################################################

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding Model Loaded")


###############################################################
# Load Chroma Database
###############################################################

print("\nLoading Chroma Database...")

vectordb = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model,
)

print("Vector Database Loaded")


###############################################################
# Retriever
###############################################################

retriever = vectordb.as_retriever(
    search_kwargs={"k": TOP_K}
)


###############################################################
# Prompt
###############################################################

prompt = PromptTemplate.from_template(
"""
You are an expert Pharmacovigilance Medical Reviewer.

You MUST answer ONLY using the supplied context.

Do NOT hallucinate.

If the answer is unavailable, reply exactly:

"I cannot find evidence in the retrieved documents."

---------------------------------------------------------

Retrieved Context

{context}

---------------------------------------------------------

User Question

{question}

---------------------------------------------------------

Generate your answer using the following format.

## Case Summary

## Patient Information

## Suspected Drug

## Adverse Event

## Seriousness Assessment

## Causality Assessment

## Regulatory Considerations

## Supporting Evidence

Use bullet points whenever appropriate.
"""
)


###############################################################
# Helper Function
###############################################################

def build_context(retrieved_docs):
    """
    Converts retrieved LangChain Documents into one context string.
    """

    context = []

    for index, doc in enumerate(retrieved_docs, start=1):

        context.append(
            f"""
==============================
DOCUMENT {index}

Source:
{doc.metadata}

Content:

{doc.page_content}
"""
        )

    return "\n".join(context)


###############################################################
# Format Retrieved Documents
###############################################################

def format_documents(retrieved_docs):
    """
    Formats documents for Streamlit display.
    """

    formatted_docs = []

    for doc in retrieved_docs:

        formatted_docs.append(
            {
                "metadata": doc.metadata,
                "content": doc.page_content,
            }
        )

    return formatted_docs


###############################################################
# Retrieve Documents
###############################################################

def retrieve_documents(question):
    """
    Retrieve the top-k most relevant documents along with
    similarity scores.
    """

    results = vectordb.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    documents = []
    scores = []

    for document, score in results:

        documents.append(document)

        scores.append(score)

    return documents, scores


###############################################################
# Prepare Context
###############################################################

def prepare_context(question):
    """
    Retrieves documents once and prepares the shared context.
    """

    retrieved_docs, similarity_scores = retrieve_documents(question)

    context = build_context(retrieved_docs)

    return {
        "context": context,
        "documents": retrieved_docs,
        "formatted_documents": format_documents(retrieved_docs),
        "scores": similarity_scores,
    }


###############################################################
# Everything below this line will load the LLMs
###############################################################

###############################################################
# LLM Imports
###############################################################

# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     AutoModelForSeq2SeqLM,
#     pipeline,
# )

# from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# import torch


# ###############################################################
# # Detect Device
# ###############################################################

# DEVICE = 0 if torch.cuda.is_available() else -1

# print("\nDevice:", "GPU" if DEVICE == 0 else "CPU")


# ###############################################################
# # LLM Registry
# ###############################################################

llms = {}


# ###############################################################
# # OpenAI GPT-4.1-mini
# ###############################################################

# try:

#     print("\nLoading GPT-4.1-mini...")

#     llms["GPT-4.1-mini"] = ChatOpenAI(

#         api_key=OPENAI_API_KEY,

#         model="gpt-4.1-mini",

#         temperature=0,

#     )

#     print("Loaded GPT-4.1-mini")

# except Exception as e:

#     print(e)

#     print("Unable to load GPT-4.1-mini")


# ###############################################################
# # OpenAI GPT-4.1-nano
# ###############################################################

# try:

#     print("\nLoading GPT-4.1-nano...")

#     llms["GPT-4.1-nano"] = ChatOpenAI(

#         api_key=OPENAI_API_KEY,

#         model="gpt-4.1-nano",

#         temperature=0,

#     )

#     print("Loaded GPT-4.1-nano")

# except Exception as e:

#     print(e)

#     print("Unable to load GPT-4.1-nano")


# ###############################################################
# # FLAN-T5
# ###############################################################

# try:

#     print("\nLoading FLAN-T5...")

#     flan_pipe = pipeline(

#         task="text2text-generation",

#         model="google/flan-t5-large",

#         max_new_tokens=512,

#         device=DEVICE,

#     )

#     llms["FLAN-T5"] = HuggingFacePipeline(

#         pipeline=flan_pipe

#     )

#     print("Loaded FLAN-T5")

# except Exception as e:

#     print(e)

#     print("Unable to load FLAN-T5")


# ###############################################################
# # Phi-3 Mini
# ###############################################################

# try:

#     print("\nLoading Phi-3 Mini...")

#     phi_pipe = pipeline(

#         task="text-generation",

#         model="microsoft/Phi-3-mini-4k-instruct",

#         trust_remote_code=True,

#         max_new_tokens=512,

#         device=DEVICE,

#     )

#     llms["Phi-3 Mini"] = HuggingFacePipeline(

#         pipeline=phi_pipe

#     )

#     print("Loaded Phi-3 Mini")

# except Exception as e:

#     print(e)

#     print("Unable to load Phi-3 Mini")

###############################################################
# LLM Registry
###############################################################

llms = {}

###############################################################
# GPT-4.1-mini
###############################################################

try:

    print("\nLoading GPT-4.1-mini...")

    llms["GPT-4.1-mini"] = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4.1-mini",
        temperature=0,
    )

    print("Loaded GPT-4.1-mini")

except Exception as e:

    print(e)

###############################################################
# GPT-4.1-nano
###############################################################

try:

    print("\nLoading GPT-4.1-nano...")

    llms["GPT-4.1-nano"] = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4.1-nano",
        temperature=0,
    )

    print("Loaded GPT-4.1-nano")

except Exception as e:

    print(e)

###############################################################
# Llama 3.2 (Ollama)
###############################################################

try:

    print("\nLoading Llama 3.2...")

    llms["Llama 3.2"] = ChatOllama(

        model="llama3.2",

        temperature=0,

    )

    print("Loaded Llama 3.2")

except Exception as e:

    print(e)

###############################################################
# Qwen 2.5 (Ollama)
###############################################################

try:

    print("\nLoading Qwen 2.5...")

    llms["Qwen 2.5"] = ChatOllama(

        model="qwen2.5",

        temperature=0,

    )

    print("Loaded Qwen 2.5")

except Exception as e:

    print(e)

###############################################################
# Loaded Models
###############################################################

print("\nLoaded Models")

for model in llms:

    print("✓", model)



print()


def generate_response(llm, prompt_text):
    """
    Generate a response from any LangChain LLM.
    """

    try:

        response = llm.invoke(prompt_text)

        if hasattr(response, "content"):
            return response.content

        return str(response)

    except Exception as e:

        return f"Error: {str(e)}"


def build_parallel_chain(prompt_text):
    """
    Creates a RunnableParallel that sends the same prompt
    to every loaded LLM.
    """

    runnables = {}

    for model_name, llm in llms.items():

        runnables[model_name] = RunnableLambda(
            lambda _, llm=llm: generate_response(
                llm,
                prompt_text
            )
        )

    return RunnableParallel(**runnables)





def ask_all_models(question):

    ##########################################################
    # Retrieve documents once
    ##########################################################

    retrieved_docs, similarity_scores = retrieve_documents(question)

    ##########################################################
    # Build shared context
    ##########################################################

    context = build_context(retrieved_docs)

    ##########################################################
    # Format prompt once
    ##########################################################

    prompt_text = prompt.format(

        context=context,

        question=question,

    )

    ##########################################################
    # Run every LLM in parallel
    ##########################################################

    parallel_chain = build_parallel_chain(prompt_text)

    outputs = parallel_chain.invoke({})

    ##########################################################
    # Return everything Streamlit needs
    ##########################################################

    return {

        "question": question,

        "context": context,

        "answers": outputs,

        # "documents": retrieved_docs,
        "documents": format_documents(retrieved_docs),

        "formatted_documents": format_documents(retrieved_docs),

        "scores": similarity_scores,

    }








###############################################################
# Generate Response
###############################################################


    



if __name__ == "__main__":

    question = input("Ask a question: ")

    result = ask_all_models(question)

    print("\n")

    print("=" * 80)
    print("LLM RESPONSES")
    print("=" * 80)

    for model, answer in result["answers"].items():

        print("\n")
        print("=" * 80)
        print(model)
        print("=" * 80)

        print(answer)

    print("\n")
    print("=" * 80)
    print("RETRIEVED DOCUMENTS")
    print("=" * 80)

    for score, doc in zip(result["scores"], result["documents"]):

        print()

        print("Similarity Score:", score)

        # print(doc.metadata)

        # print(doc.page_content[:500])

        print(doc["metadata"])

        print(doc["content"][:500])

        print("-" * 80)

        # PV0081751