from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import json
import streamlit as st
import os
from PIL import Image

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.schema import LLMResult
from langchain.schema.messages import SystemMessage

from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# a modifier pour mon use case

questions = [
    "Gestion du Contrat : Comment gérer les modifications du contrat, telles que la diminution du montant d'assurance ou les changements liés à la consommation de tabac ?",
    "Résiliation du Contrat : Quelles sont les conditions et les conséquences d'une résiliation du contrat ?",
    "Capital-Décès : Sous quelles conditions le capital-décès est-il versé ? Quelles sont les implications d'un capital-décès réduit ?",
    "Coût de l'Assurance : Comment est calculé le coût de l'assurance (CDA) et comment affecte-t-il la valeur du contrat ?",
]


def prepare_file(uploaded_file):
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


def rag_tool_openai(filename: str):
    loader = PyPDFLoader(filename)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "search_in_document",
        "Searches and returns documents.",
    )
    tools = [tool]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    context = """
Tu es un assistant interactif et personnalisé pour la gestion de contrats d'assurance maladie.

Assistant Assurance Maladie : est une IA conçue pour être un expert en gestion et en optimisation de contrats d'assurance maladie, spécialisée dans l'utilisation et la compréhension des produits d'assurance santé pour les particuliers et les professionnels. Cette IA maîtrise les aspects réglementaires, les conditions de couverture, ainsi que les stratégies d'optimisation des contrats d'assurance. Pour chaque question posée par l'utilisateur concernant son contrat d'assurance maladie, le GPT répondra en posant une question en retour, pour personnaliser sa réponse en fonction des informations fournies par l'utilisateur. Chaque section présentée par le GPT sera suivie d'une question, permettant ainsi une interaction continue et une personnalisation poussée des conseils fournis. Cette méthode garantit que chaque conseil est spécifiquement adapté aux besoins et objectifs de l'utilisateur en matière d'assurance maladie.
Pour toute question sur les formations ou l'approfondissement des connaissances en assurance, tu proposeras le lien suivant : [Association Française de l'Assurance](https://www.ffa-assurance.fr/) parmi les résultats.

Tu peux chercher les informations dans le document d'assurance fourni par l'utilisateur.

    """
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm, tools, system_message=sys_message, verbose=True
    )

    return agent_executor


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Le Co-pilot de ton contrat d'assurance")

st.markdown(
    """
<style>.element-container:has(#button-after) + div button {
    height: 150px;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    backgroundColor: #573666;
    textColor: #ffffff;
 }</style>""",
    unsafe_allow_html=True,
)
# changer les adresses de mes logos

img_col0, img_col1 = st.columns(2)
img_col0.image(Image.open("static/mozza.png"))
img_col1.image(Image.open("static/logo_seyna_150_orange.png"))
st.title("Le Co-pilot de ton contrat d'assurance")

st.write("Merci de télécharger votre contrat d'assurance PDF")

file = st.file_uploader("Upload a pdf", type="pdf")
if file is not None and "agent" not in st.session_state:
    file_path = prepare_file(file)
    st.session_state.agent = rag_tool_openai(file_path)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
if "agent" in st.session_state and "start" not in st.session_state:
    cols = st.columns(int(len(questions) / 2))
    for i, question in enumerate(questions):
        if cols[int(i / 2)].button(question):
            st.session_state.start = True
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})
            response = st.session_state.agent({"input": question})["output"]
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

response = ""
# React to user input
if "agent" in st.session_state:
    if prompt := st.chat_input("Encore une question ?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = st.session_state.agent({"input": prompt})["output"]

# Display assistant response in chat message container
if "agent" in st.session_state:
    with st.chat_message("assistant"):
        st.markdown(response)

# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
