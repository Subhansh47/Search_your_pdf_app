import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Model and tokenizer loading
checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
st.cache_resource()
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
st.cache_resource()
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']

    # Print the keys of generated_text to understand its structure
    # st.write("Keys in generated_text:", generated_text.keys())
    # st.write(generated_text['query'])
    # st.write(generated_text['result'])
    # st.write(generated_text['source_documents'])

    # Check if 'metadata' exists in generated_text
    metadata = generated_text.get('metadata', None)
    if metadata is not None:
        # Continue processing metadata
        st.write("Metadata:", metadata)
    else:
        st.write("No metadata found in generated_text.")

    # wrapped_text = textwrap.fill(answer, 100)
    return answer, metadata


def main():
    st.title("Welcome to Search Your PDF App")
    question = st.text_input("Enter your Question:")
    if st.button("Ask"):
        ans, meta = process_answer(question)
        st.write("Your Answer:")
        st.write(ans)

if __name__ == '__main__':
    main()
