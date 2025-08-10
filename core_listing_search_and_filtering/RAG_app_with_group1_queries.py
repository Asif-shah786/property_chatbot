####################################################################
#                         import
####################################################################

import warnings
import logging
from chromadb.config import Settings as ChromaSettings
from chromadb import PersistentClient

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="Warning: model not found. Using cl100k_base encoding."
)
warnings.filterwarnings("ignore", message=".*model not found.*")
warnings.filterwarnings("ignore", message=".*cl100k_base.*")

# Reduce noisy library logs (e.g., tiktoken/cl100k_base warnings issued via logger)
for _logger in ["langchain", "langchain_openai", "langchain_community"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)

import os, glob, shutil
import math, json, re, sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Azure OpenAI only
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory


from langchain.schema import format_document


# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Remove Cohere/HuggingFace providers

# Import streamlit
import streamlit as st

####################################################################
#              Config: LLM services, assistant language,...
####################################################################
list_LLM_providers = [":rainbow[**Azure OpenAI**]"]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Contextual compression",
    "Vectorstore backed retriever",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)

####################################################################
#            Create app interface with streamlit
####################################################################
st.set_page_config(page_title="Chat With Your Data")

st.title("Santa - Your Property Advisor")

# API keys (set only once to preserve persistence across reruns)
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""
if "cohere_api_key" not in st.session_state:
    st.session_state.cohere_api_key = ""
if "hf_api_key" not in st.session_state:
    st.session_state.hf_api_key = ""


def initialize_session_from_env():
    """Load .env values into Streamlit session_state once and keep them persistent."""
    if st.session_state.get("_env_initialized"):
        return

    # Load from environment
    azure_key_env = os.getenv("azureKey", "")
    google_key_env = os.getenv("googleapikey", "")
    cohere_key_env = os.getenv("cohereapikey", "")
    default_vs_path_env = os.getenv("defaultVectorStorePath", "")

    # Persist into session_state (do not overwrite existing non-empty values)
    if not st.session_state.get("azure_api_key") and azure_key_env:
        st.session_state.azure_api_key = azure_key_env
    if not st.session_state.get("google_api_key") and google_key_env:
        st.session_state.google_api_key = google_key_env
    if not st.session_state.get("cohere_api_key") and cohere_key_env:
        st.session_state.cohere_api_key = cohere_key_env
    if not st.session_state.get("default_vectorstore_path") and default_vs_path_env:
        st.session_state.default_vectorstore_path = default_vs_path_env

    # Defaults for app behavior
    if "LLM_provider" not in st.session_state:
        st.session_state.LLM_provider = "Azure OpenAI"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-5-mini"
    if "temperature" not in st.session_state:
        st.session_state.temperature = None  # Azure: no temperature control
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.95
    if "assistant_language" not in st.session_state:
        st.session_state.assistant_language = "english"

    st.session_state._env_initialized = True


def expander_model_parameters(
    LLM_provider="Azure OpenAI",
    text_input_API_key="Azure OpenAI API Key - [Get an API key](https://portal.azure.com/)",
    list_models=["gpt-5-mini", "gpt-4", "gpt-3.5-turbo"],
    default_api_key="",
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "Azure OpenAI":
        st.session_state.azure_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your Azure API key",
            value=default_api_key,
        )
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
            value=default_api_key,
        )
        st.session_state.azure_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "HuggingFace":
        st.session_state.hf_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.azure_api_key = ""
        st.session_state.google_api_key = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", list_models
        )

        # Parameters (Azure: hide temperature, keep top_p for completeness)
        st.session_state.temperature = None
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )


def sidebar_and_documentChooser():
    """Create the sidebar and the a tabbed pane: the first tab contains a document chooser (create a new vectorstore);
    the second contains a vectorstore chooser (open an old vectorstore)."""

    with st.sidebar:
        st.caption("RAG Chatbot — Azure OpenAI + Chroma")

        # Set default provider to Azure OpenAI
        if "LLM_provider" not in st.session_state:
            st.session_state.LLM_provider = "Azure OpenAI"

        # Azure only: no provider switcher
        llm_chooser = list_LLM_providers[0]

        st.divider()
        # Pre-fill Azure API key from session (persisted) or environment
        default_azure_key = st.session_state.get("azure_api_key") or os.getenv(
            "azureKey", ""
        )
        expander_model_parameters(
            LLM_provider="Azure OpenAI",
            text_input_API_key="Azure OpenAI API Key - [Get an API key](https://portal.azure.com/)",
            list_models=[
                "gpt-5-mini",
                "gpt-4",
                "gpt-3.5-turbo",
            ],
            default_api_key=default_azure_key,
        )
        # Assistant language
        st.session_state.assistant_language = st.selectbox(
            "Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retriever")
        st.session_state.retriever_type = st.selectbox(
            "Select retriever type", list_retriever_types
        )

        st.write("\n\n")
        st.write(
            "ℹ _Your Azure OpenAI API key, model parameters, and retriever choice are used when loading or creating a vectorstore._"
        )

        st.divider()
        # Vectorstore status and actions (moved from main to sidebar)
        st.subheader("📚 Vectorstore")
        default_vs_path = st.session_state.get("default_vectorstore_path")
        if st.session_state.get("vectorstore_loaded"):
            st.success("Status: ready")
            if default_vs_path:
                st.caption(f"Using: {default_vs_path}")
        else:
            st.info("No existing vectorstore detected.")
            if st.button(
                "Create Manchester Properties Vectorstore",
                type="primary",
                key="btn_create_vs_sidebar",
            ):
                create_manchester_vectorstore()

        # Display error messages if any
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except Exception:
            pass

    # Auto-load the default vectorstore
    if "vectorstore_loaded" not in st.session_state:
        st.session_state.vectorstore_loaded = auto_load_default_vectorstore()

    # Main area: no vectorstore section anymore


####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################
def create_manchester_vectorstore():
    """Create a vectorstore from the Manchester properties CSV file."""

    # Progress container
    progress_container = st.container()
    status_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Step 1: Check API keys
        status_text.text("🔍 Checking API keys...")
        progress_bar.progress(10)

        error_messages = []
        if not st.session_state.get("azure_api_key"):
            error_messages.append("insert your Azure OpenAI API key")
        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
            st.warning(st.session_state.error_message)
            return
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
            st.warning(st.session_state.error_message)
            return

        # Step 2: Load CSV file
        status_text.text("📄 Loading Manchester properties CSV file...")
        progress_bar.progress(25)

        csv_path = "data/manchester_properties_for_sale.csv"
        if not os.path.exists(csv_path):
            st.error(f"Manchester properties CSV file not found at {csv_path}")
            return

        # Load CSV with Langchain
        csv_loader = CSVLoader(file_path=csv_path, encoding="utf8")
        documents = csv_loader.load()

        status_text.text(f"✅ Loaded {len(documents)} documents from CSV")

        # Step 3: Split documents to chunks
        status_text.text("✂️ Splitting documents into chunks...")
        progress_bar.progress(40)

        chunks = split_documents_to_chunks(documents)
        status_text.text(f"✅ Created {len(chunks)} text chunks")

        # Step 4: Initialize embeddings
        status_text.text("🧠 Initializing embedding model...")
        progress_bar.progress(55)

        embeddings = select_embeddings_model()
        status_text.text("✅ Embedding model ready")

        # Step 5: Create vectorstore (batched embeddings to avoid long blocking calls)
        status_text.text("🗄️ Creating vectorstore (embedding and persisting)...")
        progress_bar.progress(70)

        persist_directory = LOCAL_VECTOR_STORE_DIR.as_posix() + "/manchester_properties"
        os.makedirs(persist_directory, exist_ok=True)

        # Quick embedding connectivity test (fast fail)
        status_text.text("🔌 Testing embedding connectivity...")
        try:
            _ = embeddings.embed_query("healthcheck")
        except Exception as embed_err:
            st.error(f"❌ Embeddings call failed: {str(embed_err)}")
            return

        # Create VS first, then add in batches
        # Use explicit PersistentClient to avoid default_tenant issues and ensure on-disk persistence
        client = PersistentClient(path=persist_directory)
        vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        # Increase batch size 3x (from 64 -> 192)
        batch_size = 200
        total = len(chunks)
        added = 0
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            vector_store.add_documents(batch)
            added += len(batch)
            # Update progress between 72 and 88 as we add batches
            pct = 72 + int(16 * (added / max(total, 1)))
            progress_bar.progress(min(pct, 88))
            status_text.text(
                f"🗄️ Creating vectorstore... embedded {added}/{total} chunks"
            )

        # Persist to disk
        try:
            vector_store.persist()
        except Exception:
            pass

        # Verify on-disk persistence
        try:
            files_now = os.listdir(persist_directory)
            status_text.text(
                f"📦 Vectorstore files: {', '.join(files_now) if files_now else '(empty)'}"
            )
        except Exception:
            pass
        st.session_state.vector_store = vector_store

        status_text.text("✅ Vectorstore created successfully")

        # Step 6: Create retriever
        status_text.text("🔍 Setting up retriever...")
        progress_bar.progress(85)

        st.session_state.retriever = create_retriever(
            vector_store=st.session_state.vector_store,
            embeddings=embeddings,
            retriever_type=st.session_state.get(
                "retriever_type", "Contextual compression"
            ),
            base_retriever_search_type="similarity",
            base_retriever_k=16,
            compression_retriever_k=20,
        )

        status_text.text("✅ Retriever configured")

        # Step 7: Create conversation chain
        status_text.text("💬 Setting up conversation chain...")
        progress_bar.progress(95)

        (
            st.session_state.chain,
            st.session_state.memory,
        ) = create_ConversationalRetrievalChain(
            retriever=st.session_state.retriever,
            chain_type="stuff",
            language=st.session_state.get("assistant_language", "english"),
        )

        # Clear chat history
        clear_chat_history()

        # Step 8: Complete
        status_text.text("🎉 Finalizing setup...")
        progress_bar.progress(100)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show success message
        with status_container:
            st.success("🎉 **Manchester Properties vectorstore created successfully!**")
            st.info("📊 **Vectorstore Details:**")
            st.write(f"• **Documents loaded:** {len(documents)}")
            st.write(f"• **Text chunks created:** {len(chunks)}")
            st.write(f"• **Storage location:** `{persist_directory}`")
            st.write(f"• **Embedding model:** text-embedding-3-small")
            st.write(
                f"• **Retriever type:** {st.session_state.get('retriever_type', 'Contextual compression')}"
            )
            # Show on-disk file list
            try:
                file_list = os.listdir(persist_directory)
                st.write(
                    f"• **Files present:** {', '.join(file_list) if file_list else '(empty)'}"
                )
                # Show current vector count
                try:
                    count = st.session_state.vector_store._collection.count()  # type: ignore[attr-defined]
                    st.write(f"• **Vectors stored:** {count}")
                except Exception:
                    pass
            except Exception:
                pass
            st.write("💬 **Ready to chat!**")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ **Error creating Manchester Properties vectorstore:** {str(e)}")
        st.info("Please check your API keys and try again.")


def delte_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def langchain_document_loader():
    """
    Create document loaders for PDF, TXT and CSV files.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """

    documents = []

    try:
        txt_loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents.extend(txt_loader.load())
    except Exception as e:
        st.warning(f"Error loading TXT files: {str(e)}")

    try:
        pdf_loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,  # type: ignore[arg-type]
            show_progress=True,
        )
        documents.extend(pdf_loader.load())
    except Exception as e:
        st.warning(f"Error loading PDF files: {str(e)}")

    try:
        csv_loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob="**/*.csv",
            loader_cls=CSVLoader,  # type: ignore[arg-type]
            show_progress=True,
            loader_kwargs={"encoding": "utf8"},
        )
        documents.extend(csv_loader.load())
    except Exception as e:
        st.warning(f"Error loading CSV files: {str(e)}")

    try:
        doc_loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,  # type: ignore[arg-type]
            show_progress=True,
        )
        documents.extend(doc_loader.load())
    except Exception as e:
        st.warning(f"Error loading DOCX files: {str(e)}")

    return documents


def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model():
    """Return Azure OpenAI embeddings, with an SDK healthcheck mirroring Azure's sample code."""
    endpoint = "https://asha-me2poe2i-eastus2.cognitiveservices.azure.com/"
    embeddings_api_version = "2024-02-01"  # embeddings API version
    deployment_name = "text-embedding-3-small"  # must match your Azure deployment name

    # Guard: ensure Azure key is present
    azure_key = st.session_state.get("azure_api_key")
    if not azure_key:
        raise ValueError(
            "Azure API key not found. Please set 'azureKey' in .env or provide it in the sidebar."
        )

    # Healthcheck using AzureOpenAI SDK to mirror Azure sample behavior
    try:
        sdk_client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=endpoint,
            api_version=embeddings_api_version,
        )
        _ = sdk_client.embeddings.create(
            input=["healthcheck"],
            model=deployment_name,
        )
    except Exception as sdk_error:
        raise RuntimeError(f"Azure embeddings healthcheck failed: {sdk_error}")

    # LangChain embeddings instance used by Chroma
    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        azure_deployment=deployment_name,
        api_key=azure_key,
        api_version=embeddings_api_version,
        model=deployment_name,
    )


####################################################################
#                 HYBRID SEARCH (Group-1) INTEGRATION
####################################################################

# ======= HYBRID SEARCH CONFIG =======
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_PARQUET = Path("data/clean/manchester_properties_clean.parquet")
CLEAN_SQLITE = Path("data/clean/manchester_properties.sqlite")

# City centre (approx St Peter’s Square)
CITY_CENTRE_LAT, CITY_CENTRE_LON = 53.4780, -2.2445
DEFAULT_RADIUS_M = 2000

TEXT_FIELDS = [
    "heading",
    "summary",
    "displayAddress",
    "propertyTypeFullDescription",
    "formattedBranchName",
]


# ======= HYBRID SEARCH HELPERS =======
def ensure_clean_data_loaded():
    if "hybrid_df" not in st.session_state:
        if not CLEAN_PARQUET.exists():
            st.error("Clean parquet not found. Run the cleaning step first.")
            raise FileNotFoundError(CLEAN_PARQUET)
        df = pd.read_parquet(CLEAN_PARQUET)
        if "text_blob" not in df.columns:

            def mk_text_blob(row: pd.Series):
                parts = []
                for c in TEXT_FIELDS:
                    if c in row and pd.notna(row[c]):
                        parts.append(str(row[c]))
                return " | ".join(parts) if parts else None

            df["text_blob"] = [mk_text_blob(r) for _, r in df.iterrows()]
        st.session_state.hybrid_df = df

    if "hybrid_sql_conn" not in st.session_state:
        if not CLEAN_SQLITE.exists():
            st.error("SQLite DB not found. Run the cleaning step first.")
            raise FileNotFoundError(CLEAN_SQLITE)
        st.session_state.hybrid_sql_conn = sqlite3.connect(CLEAN_SQLITE.as_posix())


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


AREA_TOKEN = re.compile(r"\b(m\d{1,2}[a-z]?|bl\d{1,2}|wa\d{1,2})\b", re.I)


def parse_price(q: str):
    q = q.lower()
    m_between = re.search(
        r"(?:between)\s*£?\s*([\d,]+)\s*(?:and|-|to)\s*£?\s*([\d,]+)", q
    )
    m_under = re.search(r"(?:under|below|<=?)\s*£?\s*([\d,]+)", q)
    m_over = re.search(r"(?:over|>=?)\s*£?\s*([\d,]+)", q)

    def num(x):
        if not x:
            return None
        return float(x.replace(",", ""))

    if m_between:
        return num(m_between.group(1)), num(m_between.group(2))
    if m_under:
        return None, num(m_under.group(1))
    if m_over:
        return num(m_over.group(1)), None
    return None, None


def parse_beds(q: str):
    q = q.lower()
    m_range = re.search(r"(\d+)\s*-\s*(\d+)\s*beds?", q)
    m_exact = re.search(r"(\d+)\s*bed(room)?", q)
    if m_range:
        return int(m_range.group(1)), int(m_range.group(2))
    if m_exact:
        b = int(m_exact.group(1))
        return b, b
    if "studio" in q:
        return 0, 0
    return None, None


def parse_txn(q: str):
    ql = q.lower()
    if any(k in ql for k in ["rent", "to let", "let", "rental"]):
        return "rent"
    if any(k in ql for k in ["buy", "sale", "for sale", "purchase"]):
        return "buy"
    return None


def parse_flags(q: str):
    ql = q.lower()
    return {
        "auction_bool": True if "auction" in ql else None,
        "hasVirtualTour_bool": (
            True if ("virtual tour" in ql or "video tour" in ql) else None
        ),
        "hasFloorplan_bool": (
            True if ("floorplan" in ql or "floor plan" in ql) else None
        ),
        "near_city": (
            True if re.search(r"\b(city\s*centre|city\s*center)\b", ql) else None
        ),
        "student_like": True if "student" in ql else None,
    }


def parse_areas(q: str) -> List[str]:
    pcs = AREA_TOKEN.findall(q)
    return list({pc.upper() for pc in pcs})


def parse_query_to_filters(user_q: str) -> Dict[str, Any]:
    pmin, pmax = parse_price(user_q)
    bmin, bmax = parse_beds(user_q)
    txn = parse_txn(user_q)
    flags = parse_flags(user_q)
    areas = parse_areas(user_q)

    residue = user_q
    for pat in [
        r"(under|below|over|between)\s*£?\s*[\d,]+(\s*(and|-|to)\s*£?\s*[\d,]+)?",
        r"\d+\s*beds?",
        r"\d+\s*bed(room)?",
        r"studio",
        r"for sale|buy|to let|rent|rental|let",
        r"auction|virtual tour|video tour|floor ?plan",
        r"city\s*centre|city\s*center",
        r"\b(m\d{1,2}[a-z]?|bl\d{1,2}|wa\d{1,2})\b",
    ]:
        residue = re.sub(pat, " ", residue, flags=re.I)
    residue = re.sub(r"\s+", " ", residue).strip()

    return {
        "price_min": pmin,
        "price_max": pmax,
        "beds_min": bmin,
        "beds_max": bmax,
        "transactionType_norm": txn,
        "areas_outward": areas,
        **flags,
        "semantic_text": residue if residue else None,
    }


def sql_candidates(filters: Dict[str, Any], limit: int = 4000) -> pd.DataFrame:
    conn = st.session_state.hybrid_sql_conn
    wh: List[str] = []
    params: List[Any] = []
    if filters.get("transactionType_norm"):
        wh.append("transactionType_norm = ?")
        params.append(filters["transactionType_norm"])
    if filters.get("price_min") is not None:
        wh.append("price_num >= ?")
        params.append(float(filters["price_min"]))
    if filters.get("price_max") is not None:
        wh.append("price_num <= ?")
        params.append(float(filters["price_max"]))
    if filters.get("beds_min") is not None:
        wh.append("bedrooms_int >= ?")
        params.append(int(filters["beds_min"]))
    if filters.get("beds_max") is not None:
        wh.append("bedrooms_int <= ?")
        params.append(int(filters["beds_max"]))
    if filters.get("auction_bool") is True:
        wh.append("auction_bool = 1")
    if filters.get("hasVirtualTour_bool") is True:
        wh.append("hasVirtualTour_bool = 1")
    if filters.get("hasFloorplan_bool") is True:
        wh.append("hasFloorplan_bool = 1")
    if filters.get("areas_outward"):
        marks = ",".join(["?"] * len(filters["areas_outward"]))
        wh.append(f"postcode_outward IN ({marks})")
        params.extend(filters["areas_outward"])
    where_sql = "WHERE " + " AND ".join(wh) if wh else ""
    sql = f"""
        SELECT id, displayAddress, price_num, bedrooms_int, propertySubType_norm,
               transactionType_norm, latitude_num, longitude_num, text_blob
        FROM listings
        {where_sql}
        LIMIT {int(limit)}
    """
    return pd.read_sql(sql, conn, params=params)


def apply_radius_city_centre(
    df_in: pd.DataFrame, enabled: bool, radius_m: int = DEFAULT_RADIUS_M
) -> pd.DataFrame:
    if not enabled or df_in.empty:
        return df_in
    if "latitude_num" not in df_in.columns or "longitude_num" not in df_in.columns:
        return df_in.iloc[0:0]
    mask: List[bool] = []
    for lat, lon in zip(df_in["latitude_num"], df_in["longitude_num"]):
        if pd.isna(lat) or pd.isna(lon):
            mask.append(False)
        else:
            mask.append(
                haversine_m(CITY_CENTRE_LAT, CITY_CENTRE_LON, float(lat), float(lon))
                <= radius_m
            )
    return df_in.loc[mask]


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def rerank_with_embeddings_and_keywords(
    cands: pd.DataFrame, semantic_text: str, embeddings
) -> pd.DataFrame:
    if cands.empty:
        return cands
    kw_score = np.zeros(len(cands))
    if "text_blob" in cands.columns:
        try:
            tfidf = TfidfVectorizer(stop_words="english")
            X = tfidf.fit_transform(cands["text_blob"].fillna(""))
            if semantic_text:
                qv = tfidf.transform([semantic_text])
                kw_score = cosine_similarity(qv, X).ravel()
        except Exception:
            pass
    emb_score = np.zeros(len(cands))
    if semantic_text:
        q_emb = embeddings.embed_query(semantic_text)
        docs = cands["text_blob"].fillna("").tolist()
        d_embs = embeddings.embed_documents(docs)
        emb_score = np.array([cosine(q_emb, d) for d in d_embs])
    score = 0.8 * emb_score + 0.2 * kw_score if semantic_text else kw_score
    out = cands.copy()
    out["score"] = score
    return out.sort_values("score", ascending=False)


def hybrid_search(user_query: str, top_k: int = 10):
    ensure_clean_data_loaded()
    filters = parse_query_to_filters(user_query)
    cands = sql_candidates(filters, limit=4000)
    cands = apply_radius_city_centre(cands, enabled=bool(filters.get("near_city")))
    if cands.empty:
        return cands
    embeddings = select_embeddings_model()
    semantic_text = filters.get("semantic_text") or ""
    ranked = rerank_with_embeddings_and_keywords(
        cands, semantic_text=semantic_text, embeddings=embeddings
    )
    cols = [
        "id",
        "displayAddress",
        "price_num",
        "bedrooms_int",
        "propertySubType_norm",
        "transactionType_norm",
        "score",
    ]
    cols = [c for c in cols if c in ranked.columns]
    return ranked[cols].head(top_k)


def render_results_table(df: pd.DataFrame, title="Results"):
    st.subheader(title)
    if df.empty:
        st.info("No results. Try widening your budget or area.")
        return
    for _, r in df.iterrows():
        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"**£{int(r['price_num']):,}** · {int(r['bedrooms_int']) if not pd.isna(r['bedrooms_int']) else '?'} bed · {r.get('propertySubType_norm','?')}"
                )
                st.caption(r.get("displayAddress", ""))
                st.caption(f"Listing ID: `{r['id']}`")
            with c2:
                st.write("")
                st.write(f"Score: {r.get('score', 0):.3f}")
        st.divider()


def looks_like_portal_query(q: str) -> bool:
    ql = q.lower()
    hints = [
        "bed",
        "beds",
        "flat",
        "apartment",
        "house",
        "terraced",
        "semi",
        "detached",
        "under",
        "over",
        "between",
        "£",
        "rent",
        "buy",
        "for sale",
        "to let",
        "auction",
        "floorplan",
        "virtual tour",
        "m1",
        "m2",
        "m3",
        "m50",
        "bl",
        "wa",
        "city centre",
        "city center",
        "studio",
    ]
    return any(h in ql for h in hints)


def run_group1_flow(prompt: str):
    st.session_state.setdefault("last_query", None)
    st.session_state.setdefault("page_k", 10)
    if st.session_state.last_query != prompt:
        st.session_state.last_query = prompt
        st.session_state.page_k = 10
    df = hybrid_search(prompt, top_k=st.session_state.page_k)
    render_results_table(df, title="Top matches")
    cols = st.columns([1, 4, 1])
    with cols[1]:
        if st.button("Show more results", use_container_width=True):
            st.session_state.page_k = min(st.session_state.page_k + 20, 100)
            st.experimental_rerun()


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-v3.5",
    cohere_top_n=10,
):
    """
    create a retriever which can be a:
        - Vectorstore backed retriever: this is the base retriever.
        - Contextual compression retriever: We wrap the the base retriever in a ContextualCompressionRetriever.
            The compressor here is a Document Compressor Pipeline, which splits documents
            to smaller chunks, removes redundant documents, filters the top relevant documents,
            and reorder the documents so that the most relevant are at beginning / end of the list.
        - Cohere_reranker: CohereRerank endpoint is used to reorder the results based on relevance.

    Parameters:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: The most similar vectors are returned (default k = 16).

        compression_retriever_k: top k documents returned by the compression retriever, default = 20

       cohere_api_key = "<YOUR_COHERE_KEY>"
       cohere_model = "rerank-v3.5"  # or "rerank-english-v3.0" / "rerank-multilingual-v3.0"
       cohere_top_n = 10


    """

    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

    # Cohere reranker removed in Azure-only mode
    else:
        pass


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a Vectorstore-backed retriever) in a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    to smaller chunks, removes redundant documents, filters the top relevant documents,
    and reorder the documents so that the most relevant are at beginning / end of the list.

    Parameters:
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.
        base_retriever: a Vectorstore-backed retriever.
        chunk_size (int): Docs will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500.
        k (int): top k relevant documents to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : similarity_threshold of the  EmbeddingsFilter. default =None
    """

    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents

    # Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return compression_retriever


def CohereRerank_retriever(*args, **kwargs):
    """Cohere reranker disabled in Azure-only build."""
    raise NotImplementedError("Cohere reranker is not available in Azure-only mode")


def chain_RAG_blocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Converstaional Retreival chain.
    """

    # Progress container
    progress_container = st.container()
    status_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Step 1: Check inputs
        status_text.text("🔍 Checking inputs and API keys...")
        progress_bar.progress(10)

        error_messages = []
        if (
            not st.session_state.get("azure_api_key")
            and not st.session_state.get("openai_api_key")
            and not st.session_state.get("google_api_key")
            and not st.session_state.get("hf_api_key")
        ):
            error_messages.append(
                f"insert your {st.session_state.get('LLM_provider', 'Azure OpenAI')} API key"
            )

        # Cohere not required in Azure-only build
        if not st.session_state.get("uploaded_file_list"):
            error_messages.append("select documents to upload")
        if st.session_state.get("vector_store_name") == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
            st.warning(st.session_state.error_message)
            return
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
            st.warning(st.session_state.error_message)
            return
        else:
            st.session_state.error_message = ""

        # Step 2: Clean up old files
        status_text.text("🧹 Cleaning up old temporary files...")
        progress_bar.progress(20)
        delte_temp_files()

        # Step 3: Upload documents
        status_text.text("📤 Uploading documents to temporary directory...")
        progress_bar.progress(30)

        if st.session_state.get("uploaded_file_list") is not None:
            uploaded_files = []
            for uploaded_file in st.session_state.uploaded_file_list:
                error_message = ""
                try:
                    temp_file_path = os.path.join(
                        TMP_DIR.as_posix(), uploaded_file.name
                    )
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    uploaded_files.append(uploaded_file.name)
                except Exception as e:
                    error_message += str(e)
                if error_message != "":
                    st.warning(f"Errors: {error_message}")

            status_text.text(
                f"✅ Uploaded {len(uploaded_files)} files: {', '.join(uploaded_files)}"
            )

        # Step 4: Load documents
        status_text.text("📄 Loading documents with LangChain...")
        progress_bar.progress(40)

        documents = langchain_document_loader()
        status_text.text(f"✅ Loaded {len(documents)} documents")

        # Step 5: Split documents
        status_text.text("✂️ Splitting documents into chunks...")
        progress_bar.progress(50)

        chunks = split_documents_to_chunks(documents)
        status_text.text(f"✅ Created {len(chunks)} text chunks")

        # Step 6: Initialize embeddings
        status_text.text("🧠 Initializing embedding model...")
        progress_bar.progress(60)

        embeddings = select_embeddings_model()
        status_text.text("✅ Embedding model ready")

        # Step 7: Create vectorstore
        status_text.text("🗄️ Creating vectorstore...")
        progress_bar.progress(70)

        vector_store_name = (
            st.session_state.get("vector_store_name") or "custom_vectorstore"
        )
        persist_directory = LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + vector_store_name
        os.makedirs(persist_directory, exist_ok=True)

        # Use PersistentClient as in the Manchester flow to avoid tenant issues
        # Build via PersistentClient as well for consistency
        client = PersistentClient(path=persist_directory)
        st.session_state.vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        # Add docs in larger batches as requested
        batch_size_custom = 192
        for i in range(0, len(chunks), batch_size_custom):
            st.session_state.vector_store.add_documents(
                chunks[i : i + batch_size_custom]
            )
        try:
            st.session_state.vector_store.persist()
        except Exception:
            pass
        # Ensure data is flushed to disk
        try:
            st.session_state.vector_store.persist()
        except Exception:
            pass
        status_text.text("✅ Vectorstore created successfully")

        # Step 8: Create retriever
        status_text.text("🔍 Setting up retriever...")
        progress_bar.progress(80)

        st.session_state.retriever = create_retriever(
            vector_store=st.session_state.vector_store,
            embeddings=embeddings,
            retriever_type=st.session_state.get(
                "retriever_type", "Contextual compression"
            ),
            base_retriever_search_type="similarity",
            base_retriever_k=16,
            compression_retriever_k=20,
        )
        status_text.text("✅ Retriever configured")

        # Step 9: Create conversation chain
        status_text.text("💬 Setting up conversation chain...")
        progress_bar.progress(90)

        (
            st.session_state.chain,
            st.session_state.memory,
        ) = create_ConversationalRetrievalChain(
            retriever=st.session_state.retriever,
            chain_type="stuff",
            language=st.session_state.get("assistant_language", "english"),
        )

        # Step 10: Clear chat history and complete
        status_text.text("🎉 Finalizing setup...")
        progress_bar.progress(95)

        clear_chat_history()

        # Step 11: Complete
        status_text.text("🎉 Setup complete!")
        progress_bar.progress(100)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show success message
        with status_container:
            st.success(
                f"🎉 **Vectorstore '{st.session_state.get('vector_store_name')}' created successfully!**"
            )
            st.info("📊 **Vectorstore Details:**")
            st.write(
                f"• **Documents uploaded:** {len(st.session_state.get('uploaded_file_list', []))}"
            )
            st.write(f"• **Documents loaded:** {len(documents)}")
            st.write(f"• **Text chunks created:** {len(chunks)}")
            st.write(f"• **Storage location:** `{persist_directory}`")
            st.write(f"• **Embedding model:** text-embedding-3-small")
            st.write(
                f"• **Retriever type:** {st.session_state.get('retriever_type', 'Contextual compression')}"
            )
            st.write("💬 **Ready to chat!**")

    except Exception as error:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ **Error creating vectorstore:** {str(error)}")
        st.info("Please check your inputs and try again.")


####################################################################
#                       Create memory
####################################################################


def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models"""

    if model_name == "gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024  # max_tokens for 'gpt-3.5-turbo' = 4096

        # Use Azure OpenAI if that's the provider, otherwise use regular OpenAI
        if st.session_state.get("LLM_provider") == "Azure OpenAI":
            memory = ConversationSummaryBufferMemory(
                max_token_limit=memory_max_token,
                llm=AzureChatOpenAI(
                    azure_deployment="gpt-5-mini",
                    azure_endpoint="https://asha-me2poe2i-eastus2.cognitiveservices.azure.com/",
                    api_key=st.session_state.azure_api_key,
                    api_version="2024-12-01-preview",
                    # gpt-5-mini doesn't support temperature parameter
                ),
                return_messages=True,
                memory_key="chat_history",
                output_key="answer",
                input_key="question",
            )
        else:
            # Fallback simple buffer when not using gpt-3.5 preset
            memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                output_key="answer",
                input_key="question",
            )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases
    the question and generates a standalone query.
    This query is then sent to the retriever, which fetches relevant documents (context)
    and passes them along with the standalone question and chat history to an LLM to answer.
    """

    # 1. Define the standalone_question prompt.
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question, in its original language.\n\n
        Chat History:\n{chat_history}\n
        Follow Up Input: {question}\n
        Standalone question:""",
    )

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents)
    # to the `LLM` wihch will answer

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    memory = create_memory(st.session_state.selected_model)

    # 4. Instantiate LLMs: standalone_query_generation_llm & response_generation_llm
    if st.session_state.LLM_provider == "Azure OpenAI":
        # Azure OpenAI configuration
        endpoint = "https://asha-me2poe2i-eastus2.cognitiveservices.azure.com/"
        deployment = "gpt-5-mini"
        api_version = "2024-12-01-preview"

        standalone_query_generation_llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=st.session_state.azure_api_key,
            api_version=api_version,
            temperature=1.0,
        )
        response_generation_llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=st.session_state.azure_api_key,
            api_version=api_version,
            temperature=1.0,
        )

    # 5. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


####################################################################
#                         Auto-load default vectorstore
####################################################################


def auto_load_default_vectorstore():
    """Auto-load the default vectorstore if it exists and API keys are available."""
    try:
        # Ensure env/session keys are initialized and persistent
        initialize_session_from_env()

        # Debug lines removed for cleaner UI

        # Initialize session state variables if they don't exist
        if "temperature" not in st.session_state:
            # Don't set temperature for Azure OpenAI
            if st.session_state.get("LLM_provider") != "Azure OpenAI":
                st.session_state.temperature = 0.5
            else:
                st.session_state.temperature = None
        if "top_p" not in st.session_state:
            st.session_state.top_p = 0.95
        if "assistant_language" not in st.session_state:
            st.session_state.assistant_language = "english"
        if "LLM_provider" not in st.session_state:
            st.session_state.LLM_provider = "Azure OpenAI"
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "gpt-5-mini"

        # Get keys from session (persisted) or environment as fallback
        azure_api_key = st.session_state.get("azure_api_key") or os.getenv("azureKey")
        cohere_api_key = st.session_state.get("cohere_api_key") or os.getenv(
            "cohereapikey"
        )
        default_vectorstore_path = st.session_state.get(
            "default_vectorstore_path"
        ) or os.getenv("defaultVectorStorePath")

        # Debug lines removed for cleaner UI

        if not default_vectorstore_path:
            st.warning("Default vectorstore path not found in environment variables")
            return False

        # Try both absolute and relative paths
        vectorstore_path = default_vectorstore_path
        if not os.path.exists(vectorstore_path):
            # Try relative path from current directory
            relative_path = os.path.join(
                os.getcwd(), "data", "vector_stores", "manchester_properties"
            )
            if os.path.exists(relative_path):
                vectorstore_path = relative_path
            else:
                st.warning(
                    f"Default vectorstore path not found: {default_vectorstore_path}"
                )
                return False

        # Ensure session state is populated (persist for UI/reruns)
        st.session_state.azure_api_key = azure_api_key
        st.session_state.cohere_api_key = cohere_api_key
        st.session_state.LLM_provider = "Azure OpenAI"
        st.session_state.selected_model = "gpt-5-mini"

        # Azure OpenAI configuration for embeddings
        endpoint = "https://asha-me2poe2i-eastus2.cognitiveservices.azure.com/"
        api_version = "2024-02-01"  # Use the correct API version for embeddings
        deployment_name = (
            "text-embedding-3-small"  # Use the correct embedding deployment
        )

        # Load the vectorstore embeddings using AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            azure_deployment=deployment_name,
            api_key=azure_api_key,
            api_version=api_version,
            model="text-embedding-3-small",  # Use the correct embedding model
        )

        client = PersistentClient(path=vectorstore_path)
        st.session_state.vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            persist_directory=vectorstore_path,
        )
        # Hide file listing from main UI for cleanliness
        try:
            st.session_state.vector_store.persist()
        except Exception:
            pass

        # Create retriever (no Cohere params in Azure-only build)
        st.session_state.retriever = create_retriever(
            vector_store=st.session_state.vector_store,
            embeddings=embeddings,
            retriever_type="Contextual compression",
            base_retriever_search_type="similarity",
            base_retriever_k=16,
            compression_retriever_k=20,
        )

        # Create memory and ConversationalRetrievalChain
        (
            st.session_state.chain,
            st.session_state.memory,
        ) = create_ConversationalRetrievalChain(
            retriever=st.session_state.retriever,
            chain_type="stuff",
            language=st.session_state.assistant_language,
        )

        # Clear chat history
        clear_chat_history()
        return True

    except Exception as e:
        st.error(f"Error loading default vectorstore: {str(e)}")
        return False


def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # 1. Invoke LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        if st.session_state.LLM_provider == "HuggingFace":
            answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]

        # 2. Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # 2.1. Display anwser:
            st.markdown(answer)

            # 2.2. Display source documents:
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Page: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"

                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)


####################################################################
#                         Chatbot
####################################################################
def chatbot():
    # Initialize session from .env once and keep values persistent
    initialize_session_from_env()

    # Auto-load default vectorstore on startup
    if "vectorstore_loaded" not in st.session_state:
        st.session_state.vectorstore_loaded = auto_load_default_vectorstore()

    sidebar_and_documentChooser()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        with st.spinner("Running..."):
            if looks_like_portal_query(prompt):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    st.markdown("Here are the best matches I found:")
                    run_group1_flow(prompt)
            else:
                get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()
