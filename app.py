import streamlit as st
from src.query import query_rag


st.set_page_config(page_title="Codenames Rules AI", page_icon="🕵️")

st.title("🕵️ Codenames Instruction Assistant")
st.markdown("Ask anything about the rules, clue-giving, or winning conditions.")

# Create the question bar
user_query = st.text_input("Ask a question:", placeholder="e.g., Can I use 'Apple' as a clue for 'Fruit'?")

if user_query:
    with st.spinner("Consulting the rulebook..."):
        try:
            response = query_rag(user_query)
            st.chat_message("assistant").write(response)
        except Exception as e:
            st.error(f"Error: {e}")