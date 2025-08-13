import os
import time
from pprint import pprint
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Hardcoded API key (replace with your actual key)
api_key = "AIzaSyDSuJXyJtCX0stLC3m3JzYM-lrNHIgX8WI"
genai.configure(api_key=api_key)

# Check available models
print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004'  # Updated model name
        try:
            # Handle both single strings and lists
            if isinstance(input, str):
                input = [input]
            
            embeddings = []
            for text in input:
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])
            return embeddings
        except Exception as e:
            print(f"Embedding error with {model}: {e}")
            # Fallback to older model
            try:
                embeddings = []
                for text in input:
                    result = genai.embed_content(
                        model='models/embedding-001',
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result["embedding"])
                return embeddings
            except Exception as e2:
                print(f"Fallback embedding error: {e2}")
                # Return dummy embeddings if all else fails
                return [[0.0] * 768 for _ in input]

class ImprovedRAGSystem:
    def __init__(self, db_name="improved_rag_db"):
        self.db_name = db_name
        self.chroma_client = chromadb.PersistentClient(path="./database/")
        self.db = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model with proper model name"""
        try:
            # Try different model names that should work
            model_names = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-pro'
            ]
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model with a simple query
                    test_response = self.model.generate_content("Hello")
                    print(f"Successfully initialized model: {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to initialize {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Could not initialize any Gemini model")
                
        except Exception as e:
            print(f"Model initialization error: {e}")
            # Use the basic genai.GenerativeModel without specifying version
            self.model = genai.GenerativeModel('gemini-pro')
    
    def extract_text_from_pdf(self, file_path):
        """Extract and clean text from PDF"""
        try:
            pdf_reader = PdfReader(file_path)
            text = ""
            
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += f"\n--- Page {i+1} ---\n" + page_text
                print(f"Extracted {len(page_text)} characters from page {i+1}")
            
            cleaned_text = self.clean_text(text)
            print(f"Total cleaned text length: {len(cleaned_text)} characters")
            
            # Print first 500 characters to verify extraction
            print("\n=== FIRST 500 CHARACTERS OF EXTRACTED TEXT ===")
            print(cleaned_text[:500])
            print("=" * 50)
            
            return cleaned_text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and preprocess extracted text"""
        # Remove unwanted characters and normalize text
        replacements = {
            'â€¢': '•',
            'â€"': '—',
            'Â©': '©',
            'Ã©': 'é',
            'Ã ': 'à',
            '_': ' ',
            '~': '',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 3:  # Keep lines with meaningful content
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_chunks(self, text, chunk_size=800, chunk_overlap=100):
        """Create optimized text chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        texts = text_splitter.create_documents([text])
        chunks = [chunk.page_content for chunk in texts]
        
        # Print chunks for debugging
        print(f"\n=== CREATED {len(chunks)} CHUNKS ===")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} (length: {len(chunk)}) ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        return chunks
    
    def create_vector_db(self, documents):
        """Create or update vector database"""
        try:
            # Try to get existing collection
            self.db = self.chroma_client.get_collection(
                name=self.db_name, 
                embedding_function=GeminiEmbeddingFunction()
            )
            print(f"Using existing database with {self.db.count()} documents")
            
            # Clear existing collection to avoid duplicates
            existing_ids = self.db.get()['ids']
            if existing_ids:
                self.db.delete(ids=existing_ids)
                print("Cleared existing documents")
                
        except:
            # Create new collection
            self.db = self.chroma_client.create_collection(
                name=self.db_name, 
                embedding_function=GeminiEmbeddingFunction()
            )
            print("Created new database")
        
        # Add documents to database
        print(f"Adding {len(documents)} documents to database...")
        
        for i, doc in tqdm(enumerate(documents), total=len(documents), desc="Adding documents"):
            try:
                self.db.add(
                    documents=[doc],
                    ids=[str(i)],
                    metadatas=[{"chunk_id": i, "source": "CV"}]
                )
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error adding document {i}: {e}")
                continue
        
        final_count = self.db.count()
        print(f"Database now contains {final_count} documents")
        return self.db
    
    def debug_extracted_data(self):
        """Debug function to see what data was actually extracted"""
        if self.db is None:
            print("Database not initialized")
            return
        
        print("\n" + "="*60)
        print("=== DEBUG: Checking extracted data ===")
        print(f"Total documents in database: {self.db.count()}")
        
        # Get all documents from the database
        all_docs = self.db.get()
        
        print("\n=== All chunks in database ===")
        for i, doc in enumerate(all_docs['documents']):
            print(f"\n--- Chunk {i+1} (length: {len(doc)}) ---")
            print(doc[:300] + "..." if len(doc) > 300 else doc)
        
        print("\n=== Testing searches ===")
        test_queries = ["Mansuba", "university", "education", "degree", "qualification"]
        
        for query in test_queries:
            print(f"\n--- Search for '{query}' ---")
            try:
                results = self.db.query(query_texts=[query], n_results=2)
                if results['documents'][0]:
                    for i, result in enumerate(results['documents'][0]):
                        print(f"Result {i+1}: {result[:150]}...")
                else:
                    print("No results found")
            except Exception as e:
                print(f"Search error: {e}")
        
        print("="*60)
    
    def get_relevant_passages(self, query, n_results=5):
        """Retrieve relevant passages with improved search"""
        if self.db is None:
            raise Exception("Database not initialized")
        
        try:
            # Try the query as-is first
            results = self.db.query(
                query_texts=[query], 
                n_results=min(n_results, self.db.count())
            )
            
            passages = results['documents'][0] if results['documents'] else []
            
            # If no results, try variations of the query
            if not passages:
                # Try individual words from the query
                words = query.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_results = self.db.query(query_texts=[word], n_results=2)
                        if word_results['documents'][0]:
                            passages.extend(word_results['documents'][0])
            
            print(f"Found {len(passages)} relevant passages for query: '{query}'")
            return passages[:n_results]  # Limit results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def create_enhanced_prompt(self, query, relevant_passages):
        """Create an enhanced prompt for better responses"""
        if not relevant_passages:
            return f"""You are an AI assistant. The user asked: "{query}"
            
            I don't have any relevant information to answer this question. Please say "I don't have that information in the provided context."
            """
        
        context = "\n\n".join([f"Context {i+1}: {passage}" for i, passage in enumerate(relevant_passages)])
        
        prompt = f"""You are an AI assistant that answers questions based on provided context from a person's CV/resume.

Question: {query}

Context Information:
{context}

Instructions:
- Answer the question based ONLY on the provided context
- If the information is not clearly in the context, say "I don't have that specific information in the provided context"
- Be specific and detailed when the information is available
- If the question asks about multiple things, address each part
- Use bullet points for lists when appropriate
- When mentioning names, make sure to spell them correctly as they appear in the context

Answer:"""
        
        return prompt
    
    def generate_response(self, query, max_retries=3):
        """Generate response with error handling and retries"""
        print(f"Processing query: '{query}'")
        
        relevant_passages = self.get_relevant_passages(query, n_results=5)
        
        if not relevant_passages:
            return "I couldn't find any relevant information for your question. Please try rephrasing or asking about something else."
        
        prompt = self.create_enhanced_prompt(query, relevant_passages)
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    return f"Sorry, I encountered an error generating the response: {e}"
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("RAG System is ready! Type 'quit' to exit.")
        print("Try asking about:")
        print("- What university did Mansuba attend?")
        print("- What are Mansuba's technical skills?") 
        print("- What is Mansuba's education background?")
        print("- Tell me about Mansuba's experience")
        print("="*60)
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'debug':
                    self.debug_extracted_data()
                    continue
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                print("\nThinking...")
                response = self.generate_response(query)
                print(f"\nAnswer: {response}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

# Usage example
def main():
    # Initialize RAG system
    rag = ImprovedRAGSystem()
    
    # Load and process PDF
    pdf_path = './data/Mansuba_CV.pdf'
    if os.path.exists(pdf_path):
        print("Extracting text from PDF...")
        text = rag.extract_text_from_pdf(pdf_path)
        
        if not text or len(text) < 100:
            print("Warning: Very little text extracted from PDF. Please check the PDF file.")
            return
        
        print("Creating text chunks...")
        chunks = rag.create_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        print("Creating vector database...")
        rag.create_vector_db(chunks)
        
        # Debug the extracted data
        rag.debug_extracted_data()
        
        # Start interactive chat
        rag.chat()
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please check the file path and try again.")
        print(f"Current working directory: {os.getcwd()}")
        print("Make sure your PDF is in the './data/' folder")

# Example usage for testing specific questions
def test_questions():
    rag = ImprovedRAGSystem()
    
    # Sample questions  
    questions = [
        "What university did Mansuba attend?",
        "What are Mansuba's technical skills?",
        "What projects has Mansuba worked on?", 
        "What is Mansuba's work experience?",
        "What programming languages does Mansuba know?",
        "Tell me about Mansuba's education",
        "What degree does Mansuba have?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.generate_response(question)
        print(f"A: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    main()
    
    # Uncomment this line to run test questions instead of interactive mode
    # test_questions()