import streamlit as st
from backend.langchain_chain import create_langchain
from backend.vector_db import create_vector_store
from utils.auth import check_login, register_user, get_user, change_password, init_db

# Initialize database 
init_db()

# Streamlit page configuration
st.set_page_config(page_title="Scholarship AI Assistant", page_icon=":robot_face:")
st.title("🎓 Scholarship AI Assistant")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Login/Register form
with st.form("auth_form"):
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    action = st.radio("Choose Action", ("Login", "Register", "Change Password", "Get User"))
    submitted = st.form_submit_button("Submit")

if submitted:
    if action == "Login":
        if check_login(username, password):
            st.success(f" Welcome, {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username

            # Initialize LangChain and vector store
            st.session_state.chain = create_langchain()
            # Dummy docs to initialize
            dummy_docs = ["Welcome to Scholarship AI!", "Example document"]
            st.session_state.vector_store = create_vector_store(dummy_docs)

        else:
            st.error(" Invalid username or password.")

    elif action == "Register":
        if register_user(username, password):
            st.success(" Registration successful!")
        else:
            st.error(" Username already exists.")

    elif action == "Change Password":
        # Use password field as new password
        if change_password(username, password):
            st.success("Password changed successfully!")
        else:
            st.error(" User not found.")

    elif action == "Get User":
        user = get_user(username)
        if user:
            st.success(f" User found: {user}")
        else:
            st.error(" User not found.")

# --- Logged-in view ---
if st.session_state.logged_in:
    st.header("📄 Upload Academic & Passport Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    documents = []
    if uploaded_files:
        for file in uploaded_files:
            # Read content only once
            content = file.read().decode("utf-8", errors="ignore")
            documents.append(content)

        # Create / update vector store
        st.session_state.vector_store = create_vector_store(documents)
        st.success(f" Uploaded {len(documents)} documents and updated vector store.")

    st.divider()

    st.header("🤖 Chat with Scholarship AI Assistant")
    user_question = st.text_input("Ask your question:")

    if st.button("Ask"):
        if user_question.strip():
            if st.session_state.chain:
                response = st.session_state.chain.run(question=user_question)
                st.info(f"**Assistant:** {response}")

                #  I will add the chat history to the session state
                st.session_state.chat_history.append("You", user_question)
                st.session_state.chat_history.append("Assistant", response)
            else:
                st.error(" AI chain not initialized.")
        else:
            st.warning(" Please enter a question to ask.")
    #Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for sender, message in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {message}") 

else:
    st.warning("🔒 Please log in to access the AI assistant features.")
