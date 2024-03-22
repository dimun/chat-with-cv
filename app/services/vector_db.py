import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader


class DocumentRetriever:
    def __init__(self, persist_directory='docs/chroma'):
        self.embedding = SentenceTransformerEmbeddings()
        self.persistent_client = chromadb.PersistentClient(
            path=persist_directory)
        self.langchain_chroma = Chroma(
            client=self.persistent_client, collection_name='pdf_docs', embedding_function=self.embedding)
        self.db = None

    def load_document(self, file):
        # load documents
        loader = UnstructuredMarkdownLoader(file)
        documents = loader.load()
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=550, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        # create vector database from data
        self.db = self.langchain_chroma.from_documents(docs, self.embedding)
        return self.db

    def retrieve_document(self, query, k=3):
        if self.db is None:
            raise ValueError(
                "No documents loaded. Please load documents first using load_document method.")
        # retrieve documents
        results = self.db.similarity_search(query, k=k)
        return list(map(lambda x: x.page_content, results))


document_retriever = DocumentRetriever()

