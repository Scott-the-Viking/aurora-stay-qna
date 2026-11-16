
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 3, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
#
# This is the enhanced hotel prompt from the notebook (graded pipeline version),
# carried into deployment so the served system matches what was evaluated.
# Rules 1-8 and the rule-10 fallback message are the course text, VERBATIM (the
# fallback string is what AUTO ANSWER / MANUAL LOOKUP routing keys on); rules
# 11-14 are the additions decided for this knowledge base.
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay Hotels' policies.
    The user is a hotel staff member who needs accurate policy information to help a guest.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent policies, fees, limits, conditions, or exceptions.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.
    11. If policy details vary by property, present the property-specific values (with property names)
        exactly as given in the retrieved information, or state that the detail is property-specific.
        Never apply one property's value to another property.
    12. Quote fees, times, limits, and durations exactly as written in the retrieved information.
        Do not round, estimate, or paraphrase specific amounts.
    13. If the query has multiple parts, answer each part clearly.
    14. End the answer with a single line naming the policy document(s) that support it,
        for example: "Sources: Room and Property Information Guide".

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 3,  # graded pipeline configuration (course default was 2)
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks