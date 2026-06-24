import asyncio
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable, uuid7
from langsmith.wrappers import wrap_gemini
import uuid

load_dotenv()

# Initialize OpenAI client
client = wrap_gemini(genai.Client(vertexai=True, project="ai-projects-500014", location="us-central1"))

# Configuration
thread_id = str(uuid7())

# Conversation history store (use a database in production)
thread_store: dict[str, list] = {}

def get_thread_history(thread_id: str) -> list:
    return thread_store.get(thread_id, [])

def save_thread_history(thread_id: str, messages: list):
    thread_store[thread_id] = messages

# Knowledge base storage (loaded on startup)
knowledge_base_docs: List[Tuple[str, str]] = []  # List of (filename, content) tuples
knowledge_base_embeddings: List[List[float]] = []  # Corresponding embeddings

system_prompt = """You are Emma, a customer support specialist for OfficeFlow Supply Co., a paper and office supplies distribution company serving small-to-medium businesses across North America.

ABOUT YOUR ROLE:
You're part of the Customer Experience team and have been with OfficeFlow for 3 years. You're known for being helpful, efficient, and genuinely caring about solving customer problems. Your manager emphasizes that every interaction is an opportunity to build trust and loyalty.

WHAT YOU CAN HELP WITH:
✓ Product Information - Answer questions about our catalog of office supplies, paper products, writing instruments, organizational tools, and desk accessories
✓ Inventory & Availability - Check current stock levels and help customers find what they need
✓ Product Recommendations - Suggest products based on customer needs, usage patterns, and budget
✓ General Inquiries - Handle questions about our company, product lines, and services

WHAT YOU CANNOT DIRECTLY HANDLE:
✗ Order Placement - While you can provide product info, actual ordering is done through our web portal or by contacting our sales team at sales@officeflow.com
✗ Order Status & Tracking - Direct customers to check their account portal or contact fulfillment@officeflow.com
✗ Returns & Refunds - These require approval from our Returns Department at returns@officeflow.com
✗ Account Changes - Billing, payment methods, and account settings must go through accounts@officeflow.com
✗ Technical Support - For website issues, direct to support@officeflow.com

YOUR COMMUNICATION STYLE:
- Warm and professional, never robotic or overly formal
- Use natural language - "I'd be happy to help" instead of "I will assist you"
- Show empathy when customers are frustrated
- Be specific and accurate with information
- If you don't know something, be honest and direct them to the right resource
- Use the customer's name if they provide it
- Keep responses concise but thorough

IMPORTANT - CHECK DATABASE FIRST:
When customers ask about products or inventory, ALWAYS check the database FIRST before asking clarifying questions. Give them useful information about what you find, rather than asking for more details upfront. For example, if a customer asks "do you have any paper?" - check what paper products are in stock and tell them what's available, don't ask "what type of paper are you looking for?"

INTERACTION GUIDELINES:
1. Always greet customers warmly and acknowledge their question
2. Ask clarifying questions only if truly necessary AFTER checking available information
3. Provide complete, accurate information about products and availability
4. If recommending products, explain why they're a good fit
5. End conversations by checking if they need anything else
6. When you can't help directly, provide the specific contact or resource they need
7. Never make up information - if you're unsure, say so and offer to connect them with someone who knows

YOUR TOOLS:
You have access to two powerful tools to help customers:

1. query_database - Use this for product-related questions:
   - Product availability and stock levels
   - Product prices and pricing information
   - Product details and specifications
   - Searching for specific items in inventory

2. search_knowledge_base - Use this for company policies and information:
   - Returns and refunds policies
   - Shipping and delivery information
   - Ordering process and payment methods
   - Store locations and contact information
   - Company background and general info
   - Business hours and holiday closures

Choose the right tool based on what the customer is asking about. For questions about specific products, use the database. For questions about policies, processes, or company information, use the knowledge base.

EXAMPLE INTERACTIONS:

Customer: "Do you have copy paper?"
You: "Yes, we do! We carry several types of copy paper. Are you looking for standard 8.5x11 inch letter size, or do you need a specific weight or finish? I can check what we have in stock."

Customer: "I need to return an order"
You: "I understand you need to process a return. While I can't handle returns directly, our Returns Department will be happy to help you. You can reach them at returns@officeflow.com or call 1-800-OFFICE-1 ext. 3. They typically respond within 4 business hours. Do you need any other information I can help with?"

Customer: "What's the best pen for signing documents?"
You: "For document signing, I'd recommend a pen with archival-quality ink that won't fade over time. Let me check what we have available that would work well for that purpose."

Remember: You represent OfficeFlow's commitment to excellent customer service. Be helpful, honest, and human in every interaction."""

@traceable(name="query_database",run_type="tool")
def query_database(query: str, db_path: str) -> str:
    """Execute SQL query against the inventory database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error: {str(e)}"

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks



async def load_knowledge_base(kb_dir: str = str(Path(__file__).resolve().parent / "knowledge_base")) -> None:
    """Load knowledge base documents and embeddings from cache or generate them."""
    global knowledge_base_docs, knowledge_base_embeddings
    import json

    kb_dir_path = Path(kb_dir)
    if not kb_dir_path.is_absolute():
        kb_dir_path = Path(__file__).resolve().parent / kb_dir_path

    kb_path = kb_dir_path.resolve() / "documents"
    cache_path = kb_dir_path.resolve() / "embeddings" / "embeddings.json"

    # Try to load from cache first
    if False and cache_path.exists():
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        knowledge_base_docs = [tuple(doc) for doc in cache_data["docs"]]
        knowledge_base_embeddings = cache_data["embeddings"]
        print(f"Knowledge base loaded from cache: {len(knowledge_base_docs)} chunks")
        return

    # Fall back to generating embeddings
    if not kb_path.exists():
        print(f"Warning: Knowledge base directory '{kb_dir}' not found")
        return
    knowledge_base_docs = []
    chunks = []
    for file_path in kb_path.glob("*.md"):
        if file_path.name == "CHUNKING_NOTES.md":
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            file_chunks = chunk_text(content)
            chunks.extend(file_chunks)
            for i, chunk in enumerate(file_chunks):
                knowledge_base_docs.append((f"{file_path.name}:chunk_{i}", chunk))

    if not chunks:
        print(f"Warning: No documents found in '{kb_dir}'")
        return

    print(f"Generating embeddings for {len(chunks)} chunks...")

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    
    knowledge_base_embeddings = response.embeddings
    print(f"Knowledge base loaded: {len(chunks)} chunks indexed")

@traceable(name="search_knowledge_base",run_type="tool")
async def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """Search knowledge base using semantic similarity."""
    if not knowledge_base_docs or not knowledge_base_embeddings:
        return "Error: Knowledge base not loaded"
    
    query_response = client.models.embed_content(
        model="text-embedding-004",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    # Generate embedding for query
    response =  query_response.embeddings[0].values
    query_embedding = response

    # Calculate cosine similarity with all documents
    similarities = []
    for i, doc_embedding in enumerate(knowledge_base_embeddings):
        similarity = np.dot(query_embedding, doc_embedding.values) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding.values)
        )
        similarities.append((i, similarity))

    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_k]

    # Format results
    results = []
    for idx, score in top_results:
        filename, content = knowledge_base_docs[idx]
        results.append(f"=== {filename} (relevance: {score:.3f}) ===\n{content}\n")
    print(results)
    return "\n".join(results)


tools = [
    types.FunctionDeclaration(
        name="query_database",
        description="SQL query to get information about our inventory for customers like products, quantities and prices.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": """SQL query to execute against the inventory database.

YOU DO NOT KNOW THE SCHEMA. ALWAYS discover it first:
1. Query 'SELECT name FROM sqlite_master WHERE type="table"' to see available tables
2. Use 'PRAGMA table_info(table_name)' to inspect columns for each table
3. Only after understanding the schema, construct your search queries""",
                }
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="search_knowledge_base",
        description="Search company knowledge base for information about policies, procedures, company info, shipping, returns, ordering, contact information, store locations, and business hours. Use this for non-product questions.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Natural language question or search query about company policies or information",
                }
            },
            "required": ["query"],
        },
    ),
]

tool_config = types.Tool(
    function_declarations=tools
)

import json

@traceable(name="Emma",metadata={"thread_id":thread_id})
async def chat(question: str) -> str:
    db_path = str(Path(__file__).parent / "inventory" / "inventory.db")

    history_messages = get_thread_history(thread_id)

    contents = []

    # System prompt
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=system_prompt)]
        )
    )

    # Previous history
    for msg in history_messages:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])]
            )
        )

    # Current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    )

    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[tool_config]
            ),
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_calls = [
            p.function_call
            for p in parts
            if getattr(p, "function_call", None)
        ]

        if not function_calls:
            final_text = response.text
            save_thread_history(thread_id, history_messages + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_text}
            ])
            return {"messages": get_thread_history(thread_id), "output": final_text}

        # Add model response containing function calls
        contents.append(candidate.content)

        function_responses = []

        for fc in function_calls:
            if fc.name == "query_database":
                result = query_database(
                    query=fc.args.get("query"),
                    db_path=db_path
                )

            elif fc.name == "search_knowledge_base":
                result = await search_knowledge_base(
                    query=fc.args.get("query")
                )

            else:
                result = f"Unknown tool: {fc.name}"

            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result}
                )
            )

        # Send tool results back to Gemini
        contents.append(
            types.Content(
                role="tool",
                parts=function_responses
            )
        )


async def main():
    print("Office Supplies Support Chat")
    print("=" * 50)
    print(f"Thread ID: {thread_id}")
    print()

    # Load knowledge base on startup
    await load_knowledge_base()
    print()
    print("Type 'quit' or 'exit' to end the conversation\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for chatting! Goodbye!")
            break

        if not user_input:
            continue

        response = await chat(user_input)
        print(f"\nAgent: {response["output"]}\n")

if __name__ == "__main__":
    asyncio.run(main())
