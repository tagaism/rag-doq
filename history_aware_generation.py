from unittest import result
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import chat

load_dotenv()

persistent_directory = "data/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# wire up AI model
model = ChatOpenAI(model="gpt-4o")

# Store conversation as list of messages
chat_history = []

def ask_question(user_question):
    print(f"\n --- You asked: {user_question} ---")

    # 1. Make question clear using conversation history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable question. Just return the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Search for: {search_question}")
    else:
        search_question = user_question
    
    # 2. Find relevant doc
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"    Doc {i}: {preview}...")
    
    # 3. Create final prompt
    formatted_docs = "\n".join([f"- {doc.page_content}" for doc in docs])
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {formatted_docs}

    Please provide a clear, helpful answer using only the information from these documents. If you can not find the answer, say I don't have enough information to answer that question base on the provided documemts.
    """

    # 4. get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers question base on provided documents and conversation ")
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # 5. Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me qestions! Type 'quit' to exit.")

    while True:
        question = input("\n Your question: ")
        if question.lower() == 'quit':
            print("Good bye!!")
            break
        
        ask_question(question)

if __name__ == "__main__":
    start_chat()
