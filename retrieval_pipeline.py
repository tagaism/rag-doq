from langchain_chroma import Chroma
from langchain_core.tools import retriever
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "data/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# temp query prompt
query = input("Ask a question: ") #"Which island does SpaceX lease for its launches in the Pacific?"
# query = "In what year did Tesla begin production of the Roadster?"

# retriever = db.as_retriever(search_kwargs={"k": 3})
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold":0.3 # Only return chunks with cosine similarty > 0.5
    }
)

relevant_docs = retriever.invoke(query)

print(f"Use query: {query}")
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


combined_input = f"""Based on following documents, please answer this question: {query}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can not find the answer, say "I don't have enough information to answer that question base on the provided documemts".
"""

model = ChatOpenAI(model="gpt-4o")

# Define message for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

#Display the full result
print("\n --- General Response ---")
print("Content only!!!!!!!!")
print(result.content)