"""
ContextAR - RAG Engine
Builds a FAISS vector store from exhibits_data.py and answers
questions using LangChain + OpenAI embeddings + GPT-4o.

Usage:
    # First run: build and save the index
    python rag_engine.py --build

    # Query
    python rag_engine.py --query "Who painted this blue swirly night scene?"
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from exhibits_data import EXHIBITS

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL = "text-embedding-3-small"   # cheap and fast
CHAT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index() -> FAISS:
    """Convert EXHIBITS list → LangChain Documents → FAISS index."""
    docs = []
    for exhibit in EXHIBITS:
        text = (
            f"Name: {exhibit['name']}\n"
            f"Artist: {exhibit.get('artist', 'Unknown')}\n"
            f"Year: {exhibit.get('year', 'Unknown')}\n"
            f"Type: {exhibit['type']}\n"
            f"Period: {exhibit['period']}\n\n"
            f"{exhibit['content'].strip()}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "id": exhibit["id"],
                "name": exhibit["name"],
                "artist": exhibit.get("artist", "Unknown"),
                "year": exhibit.get("year", "Unknown"),
                "type": exhibit["type"],
                "period": exhibit["period"],
            }
        ))

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Index built: {len(docs)} exhibits saved to '{FAISS_INDEX_PATH}/'")
    return vectorstore


def load_index() -> FAISS:
    """Load existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_build_index() -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH):
        return load_index()
    return build_index()


# ---------------------------------------------------------------------------
# QA chain
# ---------------------------------------------------------------------------

MUSEUM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable and friendly audio guide for the exhibition
"Western European Paintings, 15th–20th Century" at the Metropolitan Museum of Art.
Use the following exhibit information to answer the visitor's question.
Keep your answer concise (2–4 sentences) and engaging — speak as if addressing someone
standing in front of the painting.
If the answer is not in the context, say you're not sure but offer what you know.

Exhibit information:
{context}

Visitor question: {question}

Answer:"""
)


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": MUSEUM_PROMPT},
        return_source_documents=True,
    )


# ---------------------------------------------------------------------------
# Public API — used by server.py and context_router.py
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Singleton-style wrapper. Load once, query many times.

    Example:
        rag = RAGEngine()
        result = rag.query("What is the Rosetta Stone made of?")
        print(result["answer"])
        print(result["sources"])   # list of exhibit names used
    """

    def __init__(self):
        self._vectorstore = get_or_build_index()
        self._qa = build_qa_chain(self._vectorstore)

    def query(self, question: str, max_length: int = None) -> dict:
        """
        Answer a visitor question using the knowledge base.

        Args:
            question:   natural language question
            max_length: if set, truncates answer to this many characters
                        (used by context_router when hands are busy)

        Returns:
            {
                "answer": str,
                "sources": list[str]   # exhibit names retrieved
            }
        """
        result = self._qa.invoke({"query": question})
        answer = result["result"].strip()

        if max_length and len(answer) > max_length:
            answer = answer[:max_length].rsplit(" ", 1)[0] + "…"

        # Filter sources by L2 distance (lower = more relevant); threshold ~1.0
        scored = self._vectorstore.similarity_search_with_score(question, k=2)
        sources = [
            doc.metadata["name"]
            for doc, score in scored
            if score < 1.3   # L2 distance: < 1.3 = genuinely related
        ]
        return {"answer": answer, "sources": sources}

    def find_similar(self, exhibit_name: str, k: int = 2) -> list[str]:
        """
        Given an exhibit name (from exhibit_recognizer), find the k most
        relevant knowledge base entries.
        Returns list of exhibit names.
        """
        docs = self._vectorstore.similarity_search(exhibit_name, k=k)
        return [doc.metadata["name"] for doc in docs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextAR RAG Engine")
    parser.add_argument("--build", action="store_true", help="Rebuild the FAISS index")
    parser.add_argument("--query", type=str, help="Ask a question")
    args = parser.parse_args()

    if args.build:
        build_index()

    if args.query:
        rag = RAGEngine()
        result = rag.query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")

    if not args.build and not args.query:
        # Interactive demo
        rag = RAGEngine()
        print("ContextAR RAG — type a question, Ctrl+C to quit\n")
        while True:
            try:
                q = input("Question: ").strip()
                if not q:
                    continue
                result = rag.query(q)
                print(f"Answer : {result['answer']}")
                print(f"Sources: {', '.join(result['sources'])}\n")
            except KeyboardInterrupt:
                break
