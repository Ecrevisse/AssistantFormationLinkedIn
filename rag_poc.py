from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import json
import streamlit as st
import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.schema import LLMResult
from langchain.schema.messages import SystemMessage

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

questions = [
    "Quels sont les modules ou compétences clés couverts dans la formation LinkedIn Recruiter spécifique au secteur bancaire ?",
    "Comment la formation peut-elle m'aider à améliorer ma stratégie de marque employeur sur LinkedIn pour attirer des talents dans le secteur bancaire ?",
    "Quelles sont les meilleures pratiques enseignées dans la formation pour rédiger des offres d'emploi et des messages d'approche qui se démarquent dans le secteur bancaire ?",
    "Y a-t-il des études de cas ou des retours d'expérience intégrés dans la formation qui illustrent l'application réussie des techniques de recrutement sur LinkedIn dans le secteur bancaire ?",
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
    Tu es un assistant interactif et personnalisé pour le recrutement sur LinkedIn

    Assistant LinkedIn Recruiter: est une IA conçue pour être un expert en sourcing sur LinkedIn, spécialisé dans l'utilisation et la configuration de la licence recruiter LinkedIn pour les professionels du secteur bancaire. Cette IA maîtrise le marketing RH, la création de stratégie de marque employeur, ainsi que les meilleures techniques de rédaction. Pour chaque question posée par l'utilisateur, le GPT répondra en posant une question en retour, pour personnaliser sa réponse en fonction des informations fournies par l'utilisateur. Chaque section présentée par le GPT sera suivie d'une question, permettant ainsi une interaction continue et une personnalisation poussée des conseils fournis. Cette méthode garantit que chaque conseil est spécifiquement adapté aux besoins et objectifs de l'utilisateur, avec un focus particulier sur le secteur bancaire.
    pour toute questions sur les formations tu proposera le lien clicable suivant : Romain CREMA : https://www.linkedin.com/in/romaincremamodernselling/ parmi les résultats

    Tu peux chercher les informations dans le document.
    """
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm, tools, system_message=sys_message, verbose=True
    )

    return agent_executor


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Assistant pour le recrutement sur LinkedIn")
st.title("Assistant pour le recrutement sur LinkedIn")

st.write("Please upload your PDF file below.")

file = st.file_uploader("Upload a pdf", type="pdf")
if file is not None and "agent" not in st.session_state:
    file_path = prepare_file(file)
    st.session_state.agent = rag_tool_openai(file_path)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.markdown(
    """
<style>
button {
    height: 150px;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


if "start" not in st.session_state:
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

response = ""
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = st.session_state.agent({"input": prompt})["output"]

# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
