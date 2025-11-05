import streamlit as st
from backend import stream_rag_chain  # מייבא את ה-RAG מהקובץ שלך


# --- Streamlit App ---

st.set_page_config(page_title="Angular Assistant", layout="centered")

st.title("🅰️ Angular Helper Bot")

st.markdown("שאל כל שאלה על Angular — Signals, DI, Routing ועוד.")

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
if prompt := st.chat_input("מה תרצה לשאול על Angular?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant message (streaming)
    with st.chat_message("assistant"):
        response_stream = stream_rag_chain(prompt)

        # הצגה הדרגתית של הטקסט
        response_text = st.write_stream(response_stream)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
