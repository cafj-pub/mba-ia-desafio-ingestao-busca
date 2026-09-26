import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_provider():
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider in ("openai", "gemini"):
        return provider
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


def get_embeddings():
    if get_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"), transport="rest")


def get_llm():
    if get_provider() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"))

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite"), temperature=0, transport="rest")


def get_vector_store():
    for var in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(var):
            raise RuntimeError(f"Variável de ambiente {var} não definida.")

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def search_prompt(question=None):
    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    def retrieve_context(pergunta):
        results = store.similarity_search_with_score(pergunta, k=10)
        return "\n\n".join(doc.page_content.strip() for doc, _score in results)

    prompt = PromptTemplate(input_variables=["contexto", "pergunta"], template=PROMPT_TEMPLATE)

    chain = (
        {"contexto": RunnableLambda(retrieve_context), "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if question:
        return chain.invoke(question)
    return chain
