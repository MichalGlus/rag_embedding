from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env file
load_dotenv()

loader = WebBaseLoader([
    "https://angular.dev/guide/signals",
    "https://angular.dev/guide/signals/linked-signal",
    "https://angular.dev/guide/signals/resource"
])
docs = loader.load()

# INDEXING: SPLIT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# INDEXING: STORE
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
)

# RETRIEVAL AND GENERATION: RETRIEVAL
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# RETRIEVAL AND GENERATION: GENERATE
# Let’s put it all together into a chain that takes a question,
# retrieves relevant documents, constructs a prompt,
# passes it into a model, and parses the output.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")


def format_docs(original_docs):
    return "\n\n".join(doc.page_content for doc in original_docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def stream_rag_chain(text):
    for chunk in rag_chain.stream(text):
        yield chunk


def get_implementation_guidance(topic):
    """Get implementation guidance for completing Angular features"""
    guidance_prompt = f"איך לסיים ליישם {topic} ב-Angular? תן הנחיות מעשיות צעד אחר צעד."
    return rag_chain.invoke(guidance_prompt)

