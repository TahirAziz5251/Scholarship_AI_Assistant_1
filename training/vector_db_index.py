from app.backend.vector_db import create_vector_store

if __name__ == "__main__":
    documents = [" Scholarship policy 2025....", "Study in Turkiye...", "Application process for scholarships...", "Application process and deadlines..."]
    store = create_vector_store(documents)
    print("Document vector store created successfully!")