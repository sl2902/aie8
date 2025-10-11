"""Evaluate various retrievers  for a golden test dataset"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger
import pandas as pd
from uuid import uuid4
from operator import itemgetter
from dotenv import load_dotenv
load_dotenv()

from dataset_persistence import DatasetCache
from testset_persistence import TestDatasetCache, generate_test_dataset_cached

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import(
    ParentDocumentRetriever,
    MultiQueryRetriever,
    EnsembleRetriever,
)
import langsmith
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.storage import InMemoryStore
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient, models

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import(
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Create unique session ID for this evaluation run
EVALUATION_SESSION_ID = uuid4().hex[:8]
os.environ["LANGCHAIN_PROJECT"] = f"Advanced-Retrieval-Eval-{EVALUATION_SESSION_ID}"


file_path=f"./data/Projects_with_Domains.csv"
metadata_columns = [
     "Project Title",
      "Project Domain",
      "Secondary Domain",
      "Description",
      "Judge Comments",
      "Score",
      "Project Name",
      "Judge Score",
]

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

def load_dataset(filepath: str, metadata_columns: List[str]) -> List[Any]:
    """   
    Combine multiple fields to create longer documents for Ragas.
    Ragas requires documents with at least 100 tokens.
    """
    cache = DatasetCache(cache_dir="./dataset_cache")

    dataset = cache.load(
        filepath=filepath,
        metadata_columns=[
            'Project Title', 'Project Name', 'Project Domain',
            'Secondary Domain', 'Description', 'Judge Comments',
            'Score', 'Judge Score'
        ]
    )
    
    return dataset

def generate_test_dataset(
    generator_llm: LangchainLLMWrapper, 
    generator_embeddings: LangchainEmbeddingsWrapper,
    docs: List[Any],
    testset_size: int = 10) -> Dict[str, Any]:
    """Generate a test dataset using the generator LLM and embeddings"""
    cache = TestDatasetCache()
    dataset = generate_test_dataset_cached(
        generator_llm, generator_embeddings, docs, testset_size,
        force_regenerate=False
    )
    # generator = TestsetGenerator(
    #     llm=generator_llm,
    #     embedding_model=generator_embeddings,
    # )
    # dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    return dataset

def qdrant_vector_store(docs: List[Any]) -> Qdrant:
    """Setup a Qdrant vector store"""
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")

    return Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",
        collection_name="ragas_test_dataset",
    )

def chat_prompt_template() -> ChatPromptTemplate:
    """Create a chat prompt template"""
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)

def generate_chat_model() -> ChatOpenAI:
    """Create a chat model"""
    return ChatOpenAI(model="gpt-4.1-nano")

def naive_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    vector_store: Qdrant,
    k: int = 10
) -> Dict[str, Any]: 
    """Create a naive retriever"""
    naive_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    return naive_retriever

def bm25_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    docs: List[Any],
) -> Dict[str, Any]:
    """Create a BM25 retriever"""
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    return bm25_retriever

def contextual_compression_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    naive_retriever: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a contextual compression retriever"""
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=naive_retriever
    )

    return compression_retriever

def multiquery_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    naive_retriever: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a multi-query retriever"""
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=naive_retriever, llm=chat_model
    ) 

    return multi_query_retriever

def parent_document_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    docs: List[Any],
) -> Dict[str, Any]:
    """Create a parent document retriever"""
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)
    client = QdrantClient(location=":memory:")

    client.create_collection(
        collection_name="parent_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

    parent_document_vectorstore = QdrantVectorStore(
        collection_name="parent_documents", 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"), 
        client=client
    )

    store = InMemoryStore()

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore = parent_document_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    parent_document_retriever.add_documents(docs, ids=None)

    return parent_document_retriever

def ensemble_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    retriever_list: List[Any],
) -> Dict[str, Any]:
    """Create a ensemble retriever"""

    equal_weighting = [1/len(retriever_list)] * len(retriever_list)

    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=equal_weighting
    )

    return ensemble_retriever


def make_lcel_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI,
    retriever: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make a LCEL chain
    """
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

def cast_text(response: AIMessage) -> str:
    """Cast a response to a string"""
    return response.content if isinstance(response, AIMessage) else response

def create_evaluation_dataset(
    dataset: List[Any], 
    retriever: Dict[str, Any], 
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI,
    retriever_name: str = None,
    session_id: str = None,
) -> EvaluationDataset:
    """Create a evaluation dataset with session tracking"""
    logger.info(f"Make LCEL RAG chain for `{retriever_name}`")
    lcel_chain = make_lcel_chain(rag_prompt, chat_model, retriever)
    
    # Use context manager to add metadata to all runs
    with tracing_v2_enabled(
        project_name=os.environ.get("LANGCHAIN_PROJECT"),
        metadata={
            "retriever": retriever_name,
            "session_id": session_id or EVALUATION_SESSION_ID,
            "task": "retrieval_evaluation"
        }
    ):
        for doc in dataset:
            user_input = getattr(doc.eval_sample, "user_input", None) or \
                getattr(doc.eval_sample, "question", None)
            if user_input:
                retrieved_docs = lcel_chain.invoke({"question": user_input})
                doc.eval_sample.response = cast_text(retrieved_docs["response"])
                doc.eval_sample.retrieved_contexts = [
                    context.page_content for context in retrieved_docs["context"]
                ]
    
    evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
    
    return evaluation_dataset

def evaluate_ragas_dataset(dataset: EvaluationDataset, evaluator_llm: LangchainLLMWrapper) -> Dict[str, Any]:
    """Evaluate a RAGAS dataset with retriever-specific metrics
    
    Focus on retrieval quality metrics:
    - LLMContextRecall: Did we retrieve the reference documents?
    - ContextEntityRecall: Did we capture key entities?
    - NoiseSensitivity: Are we filtering irrelevant documents?
    
    Note: Generation metrics (Faithfulness, FactualCorrectness, ResponseRelevancy) 
    are excluded to reduce cost when comparing retrievers.
    """
    return evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),      # Primary retrieval metric
            ContextEntityRecall(),   # Entity coverage metric
            NoiseSensitivity(),      # Noise filtering metric
        ],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
        raise_exceptions=False,
    )

def get_langsmith_cost_stats(
    project_name: str, 
    retriever_name: str = None,
    session_id: str = None
) -> Dict[str, Any]:
    """Fetch cost statistics from LangSmith for a project
    
    Args:
        project_name: LangSmith project name
        retriever_name: Optional filter for specific retriever runs
        session_id: Optional filter for specific evaluation session
    
    Returns:
        Dictionary with cost statistics
    """
    client = Client()
    
    # Get all runs from the project
    runs = client.list_runs(project_name=project_name)
    
    total_cost = 0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    run_count = 0
    
    for run in runs:
        # Filter by session ID if provided (avoids duplicates from previous runs)
        if session_id:
            run_metadata = run.extra.get("metadata", {}) if hasattr(run, 'extra') and run.extra else {}
            if run_metadata.get("session_id") != session_id:
                continue
        
        # Filter by retriever name if provided
        if retriever_name:
            run_metadata = run.extra.get("metadata", {}) if hasattr(run, 'extra') and run.extra else {}
            if run_metadata.get("retriever") != retriever_name:
                continue
            
        # LangSmith stores cost info in the run metadata
        if hasattr(run, 'total_cost') and run.total_cost:
            total_cost += run.total_cost
        
        # Token usage info
        if hasattr(run, 'total_tokens') and run.total_tokens:
            total_tokens += run.total_tokens
        if hasattr(run, 'prompt_tokens') and run.prompt_tokens:
            prompt_tokens += run.prompt_tokens
        if hasattr(run, 'completion_tokens') and run.completion_tokens:
            completion_tokens += run.completion_tokens
            
        run_count += 1
    
    return {
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "run_count": run_count,
        "avg_cost_per_run": total_cost / run_count if run_count > 0 else 0,
    }


def main():
    logger.info("Loading CSV dataset")
    dataset = load_dataset(file_path, metadata_columns)
    
    logger.info("Generate test dataset")
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    ds = generate_test_dataset(generator_llm, generator_embeddings, dataset)
    logger.info(ds.to_pandas()[:2])

    df = ds.to_pandas()
    logger.info(f"Size of test dataset - {len(df)}")
    logger.info(df["synthesizer_name"].value_counts())

    logger.info("Create Qdrant vector store")
    vector_store = qdrant_vector_store(dataset)

    logger.info("Create rag prompt template")
    rag_prompt = chat_prompt_template()

    logger.info("Create chat model")
    chat_model = generate_chat_model()

    retriever_map = {
        "naive_retriever": naive_retriever_chain(rag_prompt, chat_model, vector_store),
        "bm25_retriever": bm25_retriever_chain(rag_prompt, chat_model, dataset),
        "contextual_compression_retriever": contextual_compression_retriever_chain(
            rag_prompt, chat_model, 
            naive_retriever_chain(rag_prompt, chat_model, vector_store)
        ),
        "multi_query_retriever": multiquery_retriever_chain(rag_prompt, chat_model, 
        naive_retriever_chain(rag_prompt, chat_model, vector_store)),
        "parent_document_retriever": parent_document_retriever_chain(rag_prompt, chat_model, dataset),
        "ensemble_retriever": ensemble_retriever_chain(rag_prompt, chat_model, 
        [
            naive_retriever_chain(rag_prompt, chat_model, vector_store), 
            bm25_retriever_chain(rag_prompt, chat_model, dataset), 
            contextual_compression_retriever_chain(rag_prompt, chat_model, 
            naive_retriever_chain(rag_prompt, chat_model, vector_store)), 
            multiquery_retriever_chain(rag_prompt, chat_model, 
            naive_retriever_chain(rag_prompt, chat_model, vector_store)), 
            parent_document_retriever_chain(rag_prompt, chat_model, dataset)
        ]),
    }
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
    
    # Store results for comparison
    results_summary = []
    
    for retriever_name in retriever_map:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {retriever_name}")
        logger.info(f"{'='*60}")
        
        # Track timing
        start_time = time.time()

        logger.info(f"Generate evaluation dataset for `{retriever_name}`")
        eval_ds = create_evaluation_dataset(
            ds,
            retriever_map[retriever_name],
            rag_prompt,
            chat_model,
            retriever_name=retriever_name,
            session_id=EVALUATION_SESSION_ID,
        )

        logger.info(f"Evaluate `{retriever_name}` using RAGAS")
        eval_results = evaluate_ragas_dataset(eval_ds, evaluator_llm)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Convert EvaluationResult to dict - only numeric columns
        results_df = eval_results.to_pandas()
        # Select only numeric columns for mean calculation
        numeric_cols = results_df.select_dtypes(include=['number']).columns
        results_dict = results_df[numeric_cols].mean().to_dict()
        
        # Store results
        results_summary.append({
            "retriever": retriever_name,
            "context_recall": results_dict.get("context_recall", 0),
            "context_entity_recall": results_dict.get("context_entity_recall", 0),
            "noise_sensitivity": results_dict.get("noise_sensitivity_relevant", 0),
            "latency_seconds": latency,
        })
        
        logger.info(f"Results: {results_dict}")
        logger.info(f"Latency: {latency:.2f}s")
    
    # Fetch cost data from LangSmith
    logger.info(f"\n{'='*60}")
    logger.info("FETCHING COST DATA FROM LANGSMITH")
    logger.info(f"{'='*60}")
    
    project_name = os.environ.get("LANGCHAIN_PROJECT", "Advanced-Retrieval-Evaluation")
    
    try:
        # Get overall project costs for THIS session only
        overall_cost_stats = get_langsmith_cost_stats(
            project_name, 
            session_id=EVALUATION_SESSION_ID
        )
        logger.info(f"\nOverall Session Stats (Session ID: {EVALUATION_SESSION_ID}):")
        logger.info(f"  Total Cost: ${overall_cost_stats['total_cost']:.4f}")
        logger.info(f"  Total Tokens: {overall_cost_stats['total_tokens']:,}")
        logger.info(f"  Total Runs: {overall_cost_stats['run_count']}")
        
        # Try to get per-retriever costs for THIS session
        for result in results_summary:
            retriever_name = result["retriever"]
            try:
                cost_stats = get_langsmith_cost_stats(
                    project_name, 
                    retriever_name=retriever_name,
                    session_id=EVALUATION_SESSION_ID
                )
                result["total_cost_usd"] = cost_stats["total_cost"]
                result["total_tokens"] = cost_stats["total_tokens"]
                result["num_llm_calls"] = cost_stats["run_count"]
                logger.info(f"{retriever_name} costs: ${cost_stats['total_cost']:.4f}")
            except Exception as e:
                logger.warning(f"Could not fetch cost for {retriever_name}: {e}")
                result["total_cost_usd"] = None
                result["total_tokens"] = None
                result["num_llm_calls"] = None
    except Exception as e:
        logger.warning(f"Could not fetch LangSmith costs: {e}")
        logger.info("Continuing without cost data...")
    
    # Print summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'='*60}")
    
    summary_df = pd.DataFrame(results_summary)
    logger.info(f"\n{summary_df.to_string()}")
    
    # Save results
    summary_df.to_csv("./data/retriever_evaluation_results.csv", index=False)
    logger.info("\nResults saved to: ./data/retriever_evaluation_results.csv")
    logger.info("\nFor detailed cost analysis, check LangSmith dashboard at:")
    logger.info("https://smith.langchain.com/")
    

if __name__ == "__main__":
    main()
