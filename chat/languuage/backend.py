# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


# create model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

SYSTEM_TEMPLATE = """
    אתה עוזר שפה מתמחה שמסייע ללומדי שפות.
    
    פרמטרים:
    - שפת מקור: {source_language}
    - שפת יעד: {target_language}
    - רמה: {level} (מתחיל/בינוני/מתקדם)
    - נושא: {topic}

    הוראות:
    1. ענה אך ורק על שאלות הקשורות ללימוד השפה המבוקשת.
    2. התאם את התשובות לרמת הלומד.
    3. השתמש בשפת המקור כדי להסביר מושגים בשפת היעד.
    4. התמקד בנושא הנבחר וספק תוכן רלוונטי.
    5. אם נשאלת שאלה שאינה קשורה ללימוד שפות, השב: "אני מתמחה רק בעזרה בלימוד שפות".
    
    דוגמאות לתשובות לפי רמות:
    - מתחיל: הסברים פשוטים, מילים בסיסיות, דקדוק יסודי.
    - בינוני: משפטים מורכבים יותר, ביטויים נפוצים, דקדוק מתקדם.
    - מתקדם: ניואנסים תרבותיים, ביטויים אידיומטיים, דקדוק מורכב.
    
    התחל כל תשובה בברכה בשפת היעד ובהתאם לרמה.
"""

template = ChatPromptTemplate(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)


# Building langchain chain
chain = template | llm | StrOutputParser()

def call_llm(user_input: str, source_language: str, target_language: str, level: str, topic: str, history: list):
    response = chain.invoke(
        {
            "source_language": source_language,
            "target_language": target_language,
            "level": level,
            "topic": topic,
            "user_input": user_input,
            "history": history
        }
    )
    return response