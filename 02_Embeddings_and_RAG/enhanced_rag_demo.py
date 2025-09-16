"""
Enhanced RAG System Demo
This script demonstrates a RAG system that can ingest both text documents and YouTube transcripts.
"""

import asyncio
import os
from typing import Any, List, Dict, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

# Import your existing modules
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.youtube_utils import YouTubeTranscriptLoader
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.vectordatabase import(
    cosine_similarity,
    euclidean_distance, 
    manhattan_distance
) 


class EnhancedRAGDemo:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.chat_openai = ChatOpenAI()
        self.youtube_loader = YouTubeTranscriptLoader(language='en')
        
        # RAG prompts
        self.system_prompt = SystemRolePrompt(
            """You are a knowledgeable assistant that answers questions based strictly on provided context.

            Instructions:
            - Only answer questions using information from the provided context
            - If the context doesn't contain relevant information, respond with "I don't know"
            - Be accurate and cite specific parts of the context when possible
            - Keep responses detailed and comprehensive
            - Only use the provided context. Do not use external knowledge."""
        )
        
        self.user_prompt = UserRolePrompt(
            """Context Information:
            {context}

            Question: {user_query}

            Please provide your answer based solely on the context above."""
        )


    def load_blog_documents(self, file_path: str = "data/PMarcaBlogs.txt") -> List[str]:
        """Load and split blog documents."""
        print("Loading blog documents...")
        try:
            text_loader = TextFileLoader(file_path)
            documents = text_loader.load_documents()
            
            text_splitter = CharacterTextSplitter()
            split_documents = text_splitter.split_texts(documents)
            
            print(f"Loaded {len(split_documents)} blog document chunks")
            return split_documents
        except Exception as e:
            print(f"Error loading blog documents: {e}")
            return []

    def load_youtube_transcript(self, video_url: str) -> List[str]:
        """Load and chunk YouTube transcript."""
        print(f"Loading YouTube transcript from {video_url}...")
        try:
            # Check if video is valid
            video_info = self.youtube_loader.get_video_info(video_url)
            if not video_info["valid"]:
                print(f"Video not valid: {video_info['error']}")
                return []
            
            print(f"Video {video_info['video_id']} is valid, loading transcript...")
            
            # Get transcript chunks
            chunks = self.youtube_loader.get_transcript(video_url, chunk_duration=90)
            youtube_texts = [chunk['text'] for chunk in chunks]
            
            print(f"Loaded {len(youtube_texts)} YouTube transcript chunks")
            return youtube_texts
            
        except Exception as e:
            print(f"Error loading YouTube transcript: {e}")
            return []

    async def build_vector_database(self, blog_texts: List[str], youtube_texts: List[str]) -> None:
        """Build vector database with both blog and YouTube content."""
        print("Building vector database...")
        
        # Add blog documents
        if blog_texts:
            print("Adding blog documents to vector database...")
            await self.vector_db.abuild_from_list(blog_texts, source_name="PMarcaBlogs")
        
        # Add YouTube transcript
        if youtube_texts:
            print("Adding YouTube transcript to vector database...")
            await self.vector_db.abuild_from_list(youtube_texts, source_name="YouTube_Video", source_type="video")
        
        # Show summary
        summary = self.vector_db.get_metadata_summary()
        print(f"\nVector Database Summary:")
        print(f"Total chunks: {summary['total_chunks']}")
        print(f"Sources: {summary['sources']}")
        print(f"Distribution: {summary['chunks_per_source']}")
    
    def search_blog_only(self, query: str, k: int = 5, distance_measure=cosine_similarity) -> None:
        """Search only in blog documents."""
        print(f"\n{'='*60}")
        print(f"BLOG-ONLY SEARCH: {query}")
        print(f"{'='*60}")
        
        results = self.vector_db.search_by_text(
            query, k=k, 
            metadata_filter={"source": "PMarcaBlogs"},
            include_metadata=True,
            distance_measure=distance_measure,
        )
        
        self._display_results(results)

    def search_youtube_only(self, query: str, k: int = 5, distance_measure=cosine_similarity) -> None:
        """Search only in YouTube transcript."""
        print(f"\n{'='*60}")
        print(f"YOUTUBE-ONLY SEARCH: {query}")
        print(f"{'='*60}")
        
        results = self.vector_db.search_by_text(
            query, k=k,
            metadata_filter={"source": "YouTube_Video"},
            include_metadata=True,
            distance_measure=distance_measure,
        )
        
        self._display_results(results)

    def run_rag_query_filtered(self, query: str, source_filter: str, k: int = 4, distance_measure=cosine_similarity) -> str:
        """Run RAG query filtered by source."""
        filter_map = {
            "blog": {"source": "PMarcaBlogs"},
            "youtube": {"source": "YouTube_Video"},
            "text": {"source_type": "text"},
            "video": {"source_type": "youtube"}
        }
        
        metadata_filter = filter_map.get(source_filter.lower())
        if not metadata_filter:
            print(f"Invalid filter: {source_filter}. Using no filter.")
            metadata_filter = None
        
        print(f"\n{'='*60}")
        print(f"FILTERED RAG QUERY ({source_filter.upper()}): {query}")
        print(f"{'='*60}")
        
        # Get relevant contexts with filter
        context_list = self.vector_db.search_by_text(
            query, k=k, 
            metadata_filter=metadata_filter, 
            include_metadata=True,
            distance_measure=distance_measure
        )
        
        # Rest same as regular RAG query...
        context_prompt = ""
        for i, (context, score, metadata) in enumerate(context_list, 1):
            source_name = metadata.get('source', 'Unknown')
            chunk_index = metadata.get('chunk_index', 'N/A')
            context_prompt += f"[Source {i} - {source_name} Chunk {chunk_index}]: {context}\n\n"
        
        system_message = self.system_prompt.create_message()
        user_message = self.user_prompt.create_message(
            context=context_prompt.strip(),
            user_query=query
        )
        
        response = self.chat_openai.run([system_message, user_message])
        
        print(f"Filtered RAG Response:\n{response}")
        print(f"Sources used: {len(context_list)}")
        
        return response

    def _display_results(self, results):
        """Helper method to display search results."""
        for i, (text, score, metadata) in enumerate(results, 1):
            source_name = metadata.get('source', 'Unknown')
            chunk_index = metadata.get('chunk_index', 'N/A')
            
            print(f"\n[Result {i}] Source: {source_name} (Chunk {chunk_index})")
            print(f"Similarity Score: {score:.3f}")
            print(f"Text Preview: {text[:300]}...")
            print("-" * 50)

    def search_and_display(self, query: str, k: int = 5, distance_measure=cosine_similarity) -> None:
        """Search vector database and display results."""
        print(f"\n{'='*60}")
        print(f"SEARCH QUERY: {query}")
        print(f"{'='*60}")
        
        results = self.vector_db.search_by_text(query, k=k, include_metadata=True, distance_measure=distance_measure)
        
        for i, (text, score, metadata) in enumerate(results, 1):
            source_name = metadata.get('source', 'Unknown')
            source_type = metadata.get('source_type', 'text')
            chunk_index = metadata.get('chunk_index', 'N/A')
            
            print(f"\n[Result {i}] Source: {source_name} (Chunk {chunk_index})")
            print(f"Similarity Score: {score:.3f}")
            print(f"Text Preview: {text[:300]}...")
            print("-" * 50)

    def run_rag_query(self, query: str, k: int = 4, distance_measure=cosine_similarity) -> str:
        """Run a complete RAG query."""
        print(f"\n{'='*60}")
        print(f"RAG QUERY: {query}")
        print(f"{'='*60}")
        
        # Get relevant contexts
        context_list = self.vector_db.search_by_text(query, k=k, include_metadata=True, distance_measure=distance_measure)
        
        # Build context string
        context_prompt = ""
        for i, (context, score, metadata) in enumerate(context_list, 1):
            source_name = metadata.get('source', 'Unknown')
            chunk_index = metadata.get('chunk_index', 'N/A')
            context_prompt += f"[Source {i} - {source_name} ({source_type}) Chunk {chunk_index}, Score: {score:.3f}]: {context}\n\n"
        
        # Create messages
        system_message = self.system_prompt.create_message()
        user_message = self.user_prompt.create_message(
            context=context_prompt.strip(),
            user_query=query
        )
        
        # Get response
        response = self.chat_openai.run([system_message, user_message])
        print(f"RAG Response:\n{response}")
        print(f"\nSources used: {len(context_list)}")
        print("\nSource Details:")
        for i, (context, score, metadata) in enumerate(context_list, 1):
            source_name = metadata.get('source', 'Unknown')
            chunk_index = metadata.get('chunk_index', 'N/A')
            print(f"  [{i}] {source_name} (Chunk {chunk_index}) - Score: {score:.3f}")
    
        
        
        
        return response
    
    def run_pipeline(self, blog_file_path: str, youtube_url: str):
        """Helper method to run the ingestion, chunking and loading steps"""
        blog_texts = self.load_blog_documents(blog_file_path)
        youtube_texts = self.load_youtube_transcript(youtube_url)
        
        if not blog_texts and not youtube_texts:
            print("No content loaded. Exiting.")
            return
        
        # Build vector database
        asyncio.run(self.build_vector_database(blog_texts, youtube_texts))

    def demonstrate_system(self, blog_file_path: str, youtube_url: str):
        """Complete demonstration of the enhanced RAG system."""
        print("Enhanced RAG System Demonstration")
        print("=" * 50)
        
        # Load content
        blog_texts = self.load_blog_documents(blog_file_path)
        youtube_texts = self.load_youtube_transcript(youtube_url)
        
        if not blog_texts and not youtube_texts:
            print("No content loaded. Exiting.")
            return
        
        # Build vector database
        asyncio.run(self.build_vector_database(blog_texts, youtube_texts))
        
        # Test searches
        test_queries = [
            "What are the key insights about technology trends?",
            "What challenges are discussed?",
            "What are the main points about innovation?"
        ]
        
        print(f"\n{'='*60}")
        print("TESTING FILTERED SEARCH")
        print(f"{'='*60}")

        test_query = "What is the content about?"

        # Search all sources
        print("Search all sources")
        self.search_and_display(test_query, k=3)

        # Search only blog
        print("Search only blog")
        self.search_blog_only(test_query, k=3)

        # Search only YouTube
        print("Search only YouTube")
        self.search_youtube_only(test_query, k=3)

        # Test filtered RAG
        print(f"\n{'='*60}")
        print("TESTING FILTERED RAG")
        print(f"{'='*60}")

        rag_query = "What are the main points discussed?"
        self.run_rag_query_filtered(rag_query, "blog", k=3)
        self.run_rag_query_filtered(rag_query, "youtube", k=3)


def main():
    """Main function to run the demonstration."""
    
    # Configuration
    BLOG_FILE_PATH = "data/PMarcaBlogs.txt"
    YOUTUBE_URL = "https://www.youtube.com/watch?v=sXL1qgrPysg"
    
    # Initialize and run demo
    demo = EnhancedRAGDemo()
    demo.demonstrate_system(BLOG_FILE_PATH, YOUTUBE_URL)


if __name__ == "__main__":
    main()