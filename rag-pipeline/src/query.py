"""
Query the RAG pipeline: retrieve relevant chunks and generate a grounded response.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}

If the context does not contain enough information to answer the question, say "I don't have enough information."
"""


def query(question: str):
    # 1. Load vector store
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Retrieve relevant chunks
    results = db.similarity_search_with_score(question, k=3)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # 3. Build prompt and generate response
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(prompt)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response.content}")
    print(f"\nSources: {len(results)} chunks retrieved")


if __name__ == "__main__":
    query("What is the main topic of the document?")
