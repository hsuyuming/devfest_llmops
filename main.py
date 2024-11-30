from cachetools import TTLCache
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from model.decrypt import DecryptModel
from model.qna import QnA
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
import os 
from opentelemetry import trace 
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpOTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_community.vertex_ai_search import VertexAISearchRetriever
from util.encryption_processor import EncryptionProcessor
load_dotenv()  # take environment variables from .env.

# Opentelemetry setup
azire_endpoint = os.environ.get("ARIZE_ENDPOINT", None)
collector_endpoint = os.environ.get("COLLECTOR_ENDPOINT", None)
cloudtrace = os.environ.get("CloudTrace", False)
tracer_provider = trace_sdk.TracerProvider()
trace.set_tracer_provider(tracer_provider)

processor = EncryptionProcessor()
trace.get_tracer_provider().add_span_processor(processor)

if azire_endpoint:
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(HttpOTLPSpanExporter(azire_endpoint)))
if collector_endpoint:
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(HttpOTLPSpanExporter(collector_endpoint)))
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
if bool(cloudtrace):
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))

app = FastAPI()



ENCRYPT_KEY = APIKeyHeader(
    name="encrypt-key",
    auto_error=True,
    scheme_name="Encrypt key",
    description="Provide a valid encrypt key",
)


# Configure the cache with a TTL (e.g., 1 hour) and a maximum size
encrypt_key_cache = TTLCache(maxsize=1000, ttl=3600)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.get("/ping")
async def root():
    return {"ping": "pong"}


@app.get("/encrypt_key", description="generate encrypt key")
def generate_encrypt_key(user: str):
    try:
        encrypt_key = encrypt_key_cache.get(user)
        if encrypt_key is None:
            encrypt_key = Fernet.generate_key().decode()
            encrypt_key_cache[user] = encrypt_key
        return {"encrypt_key": encrypt_key}
    except Exception as e:  # Catch potential errors
        raise HTTPException(status_code=500, detail=f"Error generating key: {e}")


@app.post("/decrypt", description="decrypt value")
def decrypt_text(model: DecryptModel, encrypt_key: str = Security(ENCRYPT_KEY)):
    cypher = Fernet(encrypt_key)
    encrypt_text = model.encrypt_text.encode()
    decrypted_message = cypher.decrypt(encrypt_text).decode()
    return decrypted_message

@app.post("/qna", description="Q and A")
async def qna(
    model: QnA, 
    user: str, 
    encrypt_key: str = Security(ENCRYPT_KEY)
):

    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatVertexAI(model_name="gemini-pro")
    retriever = VertexAISearchRetriever(
        project_id=os.environ.get("PROJECT_ID", None),
        data_store_id=os.environ.get("DATA_STORE_ID", None),
        location_id=os.environ.get("LOCATION_ID", None),
        max_extractive_segment_count=10
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = await rag_chain.ainvoke(
        input=model.question,
        config={
            "metadata": {
                "encrypt_key": encrypt_key,
                "user": user
            }
        } 
    )
    return result

FastAPIInstrumentor.instrument_app(app)
LangChainInstrumentor().instrument()