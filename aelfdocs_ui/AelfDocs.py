import streamlit as st
st.title("AelfDocs")

# Load Settings

# Change system path to root direcotry
import os, sys, pymongo

sys.path.insert(0, '../')

ATLAS_URI = st.secrets["ATLAS_URI"]

if not ATLAS_URI:
    raise Exception ("'ATLAS_URI' is not set.  Please set it above to continue...")
else:
    print("ATLAS_URI Connection string found:", ATLAS_URI)

# Define DB variables
DB_NAME = 'aelfdocs'
COLLECTION_NAME = 'docs'
INDEX_NAME = 'idx_embedding'

# LlamaIndex will download embeddings models as needed
# Set llamaindex cache dir to ../cache dir here (Default is system tmp)
# This way, we can easily see downloaded artifacts
os.environ['LLAMA_INDEX_CACHE_DIR'] = os.path.join(os.path.abspath('../'), 'cache')

mongodb_client = pymongo.MongoClient(ATLAS_URI)

# Setup Embedding Model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Setup LLM

llm = Gemini(api_key=st.secrets["GOOGLE_API_KEY"], model="models/gemini-1.5-flash-latest")
Settings.llm = llm

# Connect Llama-index and MongoDB Atlas
vector_store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client,
                                 db_name = DB_NAME, collection_name = COLLECTION_NAME,
                                 index_name  = 'idx_embedding',
                                 )
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)

memory = ChatMemoryBuffer.from_defaults()

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        """Your name is Gandalf and you are an Aelf blockchain documentation expert"""
    )
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = memory.get_all()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt)
        response.is_dummy_stream = True
        st.write_stream(response.response_gen)
    st.session_state.messages.append({"role": "assistant", "content": response})