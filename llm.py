import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from contradiction_detector import detect_contradictions

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Primary LLM: Groq Llama, Fallback: Gemini 2.0 Flash
primary_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.0)
fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0)
llm = primary_llm.with_fallbacks([fallback_llm])

# One ChatMessageHistory per session_id
store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Prompt 1: Rewrites follow-up questions into standalone questions using chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a question reformulation assistant for a clinical trial research chatbot.\n"
     "YOUR ONLY JOB: Given the chat history and the user's latest question, "
     "rewrite the question so it is clear and standalone without needing the chat history.\n"
     "STRICT RULES:\n"
     "- Do NOT answer the question.\n"
     "- Do NOT change the meaning or intent of the question.\n"
     "- If the question is already standalone, return it exactly as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt 2: Generates the final grounded answer using retrieved trial chunks as context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a clinical trial research assistant.\n"
     "Answer the user's question using ONLY the information provided in the CONTEXT below.\n"
     "Always cite the Trial ID when using information from a specific trial.\n"
     "If the answer cannot be found in the context, say exactly:\n"
     "\"I could not find relevant information for this question in the available trials.\"\n"
     "Do not make up any information. Do not use outside knowledge.\n\n"
     "CONTEXT:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def get_answer(
    query: str,
    session_id: str,
    condition: str = None,
    drug_name: str = None,
    phase: str = None,
    top_k: int = 5,
) -> dict:
    from hybrid import hybrid_search

    try:
        session_history = get_session_history(session_id)

        # Step 1: Rewrite the question to be standalone if there is chat history
        search_query = query
        if session_history.messages:
            contextualize_messages = contextualize_q_prompt.format_messages(
                chat_history=session_history.messages,
                input=query,
            )
            context_response = llm.invoke(contextualize_messages)
            search_query = context_response.content
            print(f"Contextualized query: {search_query}")

        # Step 2: Search the database using the (possibly rewritten) query
        chunks = hybrid_search(
            query=search_query,
            condition=condition,
            drug_name=drug_name,
            phase=phase,
            top_k=top_k,
        )

        if not chunks:
            return {
                "answer":            "I could not find relevant information for this question in the available trials.",
                "chunks":            [],
                "session_id":        session_id,
                "has_contradiction": False,
                "contradictions":    [],
            }

        # Step 3: Build the context string from retrieved chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"Trial ID: {chunk['doc_id']}\n"
                f"Title: {chunk['title']}\n"
                f"Section: {chunk['section']}\n"
                f"Content: {chunk['content']}"
            )
        context = "\n\n".join(context_parts)

        # Step 4: Generate the final answer with full chat history and context
        final_messages = qa_prompt.format_messages(
            context=context,
            chat_history=session_history.messages,
            input=query,
        )
        response = llm.invoke(final_messages)
        ai_answer = response.content

        # Step 5: Check for contradictions across retrieved trials
        contradiction_result = detect_contradictions(chunks, query)

        # Step 6: Save this turn to history so the next question has context
        session_history.add_user_message(query)
        session_history.add_ai_message(ai_answer)

        return {
            "answer":            ai_answer,
            "chunks":            chunks,
            "session_id":        session_id,
            "has_contradiction": contradiction_result["has_contradiction"],
            "contradictions":    contradiction_result["contradictions"],
        }

    except Exception as e:
        print(f"LLM error: {e}")
        return {
            "answer":            "Sorry, I was unable to generate an answer. Please try again.",
            "chunks":            [],
            "session_id":        session_id,
            "has_contradiction": False,
            "contradictions":    [],
        }
