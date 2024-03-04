from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
# import arabic_reshaper
# from bidi.algorithm import get_display

loader = PyPDFLoader("Walid's Data.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(pages)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

question = "who is Walid?"
docs = vectorstore.similarity_search(question)
print(docs[0])
# result_text = str(docs[0])
# # Reshape the text
# reshaped_text = arabic_reshaper.reshape(result_text)

# # Get the display version
# display_text = get_display(reshaped_text)
# print(display_text)