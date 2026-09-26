import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_vector_store

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_SIZE = 10
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 60


def resolve_pdf_path():
    path = Path(PDF_PATH or "document.pdf")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {path}")
    return path


def is_rate_limit_error(error):
    message = str(error).lower()
    return "429" in message or "quota" in message or "rate limit" in message


def add_batch_with_retry(store, documents, ids):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(documents=documents, ids=ids)
            return
        except Exception as e:
            if not is_rate_limit_error(e) or attempt == MAX_RETRIES:
                raise
            print(f"  Limite de requisições atingido, aguardando {RETRY_WAIT_SECONDS}s (tentativa {attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_WAIT_SECONDS)


def ingest_pdf():
    pdf_path = resolve_pdf_path()
    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not splits:
        raise RuntimeError("Nenhum conteúdo extraído do PDF.")

    # Remove metadados vazios para não poluir o JSONB
    for doc in splits:
        doc.metadata = {k: v for k, v in doc.metadata.items() if v not in ("", None)}

    store = get_vector_store()

    # IDs determinísticos evitam duplicação ao reexecutar; o id é único na tabela inteira,
    # por isso inclui o nome da collection
    ids = [f"{store.collection_name}-{i}" for i in range(len(splits))]

    # Envia em lotes para respeitar os limites por minuto dos planos gratuitos
    for start in range(0, len(splits), BATCH_SIZE):
        end = start + BATCH_SIZE
        add_batch_with_retry(store, splits[start:end], ids[start:end])
        print(f"  {min(end, len(splits))}/{len(splits)} chunks processados")

    print(f"Ingestão concluída: {len(splits)} chunks de {pdf_path.name} salvos na collection '{store.collection_name}'.")


if __name__ == "__main__":
    ingest_pdf()
