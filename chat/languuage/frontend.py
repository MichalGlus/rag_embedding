import streamlit as st

from backend import call_llm

st.title("📚 Language Learning Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Learning Settings")
    
    # Source language dropdown
    source_language = st.selectbox(
        "Source Language (שפת מקור):",
        ("Hebrew", "English", "Arabic", "Russian", "Spanish", "French", "German")
    )
    
    # Target language dropdown
    target_language = st.selectbox(
        "Target Language (שפת יעד):",
        ("English", "Hebrew", "Arabic", "Russian", "Spanish", "French", "German")
    )
    
    # Level dropdown
    level = st.selectbox(
        "Level (רמה):",
        ("Beginner", "Intermediate", "Advanced")
    )
    
    # Topic dropdown
    topic = st.selectbox(
        "Topic (נושא):",
        ("General Conversation", "Business", "Travel", "Food", "Technology", "Culture", "Grammar", "Vocabulary")
    )

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything about language learning!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the response from your backend function
    with st.chat_message("assistant"):
        with st.spinner("Learning..."):
            response = call_llm(
                user_input=prompt,
                source_language=source_language,
                target_language=target_language,
                level=level,
                topic=topic,
                history=st.session_state.messages
            )
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})