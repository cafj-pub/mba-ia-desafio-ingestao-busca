# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão de um PDF em PostgreSQL + pgVector e chat via CLI que responde apenas com base no conteúdo do documento (LangChain + OpenAI ou Gemini).

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API Key da OpenAI **ou** da Google (Gemini)

## Configuração

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edite o .env e preencha OPENAI_API_KEY ou GOOGLE_API_KEY
```

Variáveis principais do `.env`:

| Variável | Descrição |
|---|---|
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | API key do provedor (pelo menos uma é obrigatória) |
| `LLM_PROVIDER` | `openai` ou `gemini` (opcional, ver [Escolha do provedor](#escolha-do-provedor)) |
| `OPENAI_EMBEDDING_MODEL` / `OPENAI_LLM_MODEL` | Modelos OpenAI (padrão: `text-embedding-3-small` / `gpt-5-nano`) |
| `GOOGLE_EMBEDDING_MODEL` / `GOOGLE_LLM_MODEL` | Modelos Gemini (padrão: `models/gemini-embedding-001` / `gemini-3.5-flash-lite`) |
| `DATABASE_URL` | Conexão com o Postgres (`postgresql+psycopg://postgres:postgres@localhost:5432/rag`) |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector |
| `PDF_PATH` | Caminho do PDF (relativo à raiz do projeto) |

### Escolha do provedor

O provedor é definido pelas API keys preenchidas e pelo `LLM_PROVIDER`:

| Keys preenchidas | `LLM_PROVIDER` | Provedor usado |
|---|---|---|
| Só `GOOGLE_API_KEY` | vazio | Gemini |
| Só `OPENAI_API_KEY` | vazio | OpenAI |
| As duas | vazio | Gemini (padrão) |
| As duas | `openai` | OpenAI |
| As duas | `gemini` | Gemini |

Ou seja, o `LLM_PROVIDER` só é necessário para usar a OpenAI quando as duas keys estão definidas.

Use sempre o mesmo provedor na ingestão e no chat: os modelos de embeddings geram vetores com dimensões diferentes, e a busca falha se a collection tiver sido criada com outro provedor (veja [Observações](#observações)).

## Execução

1. Subir o banco de dados:

```bash
docker compose up -d
```

2. Executar a ingestão do PDF (chunks de 1000 caracteres, overlap de 150):

```bash
python src/ingest.py
```

3. Rodar o chat:

```bash
python src/chat.py
```

Exemplo:

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Observações

- A ingestão pode ser executada novamente sem duplicar dados (os chunks usam IDs determinísticos).
- Ao trocar o modelo de embeddings, a dimensão dos vetores muda. Apague a collection existente (ou use outro `PG_VECTOR_COLLECTION_NAME`), ou remova o volume do banco com `docker compose down -v`, e refaça a ingestão.
