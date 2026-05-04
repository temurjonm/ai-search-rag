from openai import AsyncOpenAI

from ai_search.config import get_settings
from ai_search.schemas import Source


settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key or None)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


async def generate_answer(question: str, sources: list[Source]) -> str:
    context = "\n\n".join(
        f"[Source {index + 1}] "
        f"(document={source.filename}, chunk_id={source.chunk_id})\n"
        f"{source.excerpt}"
        for index, source in enumerate(sources)
    )

    system_prompt = (
        "You are an AI search assistant. Answer using only the provided sources. "
        "Cite claims with [Source N]. If the sources do not contain the answer, "
        "say you do not have enough information in the provided documents."
    )

    user_prompt = f"Question: {question}\n\nSources:\n{context}"

    response = await client.responses.create(
        model=settings.answer_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text