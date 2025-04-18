
import json
import os
import boto3
from langchain_aws.embeddings import BedrockEmbeddings  # Requires pip install langchain-aws
from langchain_community.llms import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Initialize AWS Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Titan Embedding Model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data ingestion function
def data_ingestion():
    print("Loading PDF documents...")
    loader = PyPDFDirectoryLoader("data")  # Ensure the PDF files are inside the 'data' folder
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Create and save FAISS vector store
def get_vector_store(docs):
    print("Creating vector embeddings...")
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    print("Vector store created successfully!")

# Initialize Llama 3 model
def get_llama3_llm():
    llm = Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )
    return llm

# Define prompt template
prompt_template = """
Human: Use the following context to answer the question. 
Provide a concise yet detailed response of at least 250 words. 
If you don't know the answer, say that you don't know.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Retrieve response from Llama 3
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    print("Chat with PDF using AWS Bedrock (Llama 3)")

    # Create vector store if needed
    if input("Do you want to create/update the vector store? (yes/no): ").strip().lower() == "yes":
        docs = data_ingestion()
        get_vector_store(docs)

    # Load FAISS index
    print("Loading vector store...")

    if not os.path.exists("faiss_index/index.faiss"):
        print("FAISS index not found! Creating new vector store...")
        docs = data_ingestion()  # Ingest data again
        get_vector_store(docs)  # Generate FAISS index

    # Now, load the FAISS index safely
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Initialize Llama 3
    llm = get_llama3_llm()

    while True:
        user_question = input("\nEnter your question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Exiting...")
            break

        print("\nProcessing your query...")
        response = get_response_llm(llm, faiss_index, user_question)
        print("\nLlama 3 Response:\n")
        print(response)

# **Fix: Correct the entry point**
if __name__ == "__main__":
    main()
