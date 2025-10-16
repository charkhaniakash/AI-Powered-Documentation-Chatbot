"""
LLM service for generating responses using Groq API.
Implements RAG (Retrieval Augmented Generation) pattern.
"""

# Import necessary libraries
import logging  # For logging
from typing import List, Dict, Any, Optional  # For type hints
import json  # For JSON operations

# Import Groq client
from groq import Groq  # Groq API client

# Internal imports
from app.config.settings import settings  # Application settings
from app.models.schemas import (  # Data models
    QueryRequest,
    QueryResponse,
    RetrievedChunk
)

# ========== Setup Logging ==========
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


# ========== LLM Service Class ==========
class LLMService:
    """
    Handles LLM interactions for generating responses.
    
    What is RAG (Retrieval Augmented Generation)?
    1. Retrieve: Find relevant context from vector database
    2. Augment: Add context to the prompt
    3. Generate: LLM generates answer based on context
    
    Benefits of RAG:
    - Reduces hallucinations (LLM has real data)
    - Keeps information up-to-date
    - Provides source citations
    - More accurate than pure LLM
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM service.
        
        Args:
            api_key: Groq API key (uses settings if None)
        """
        # Use provided API key or get from settings
        self.api_key = api_key or settings.GROQ_API_KEY
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Model configuration
        self.model_name = settings.LLM_MODEL_NAME
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        
        logger.info(
            f"LLMService initialized with model: {self.model_name}"
        )
    
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryResponse:
        """
        Generate a response using RAG pattern.
        
        This is the main method that combines:
        1. Retrieved context from vector DB
        2. User query
        3. Conversation history (optional)
        4. LLM generation
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            conversation_history: Previous messages (optional)
            
        Returns:
            QueryResponse with answer and metadata
            
        Raises:
            RuntimeError: If generation fails
        """
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # Build the prompt with context
        prompt = self._build_prompt(query, context_chunks)
        
        # Prepare messages for chat completion
        messages = self._prepare_messages(
            prompt=prompt,
            conversation_history=conversation_history
        )
        
        try:
            # Call Groq API for chat completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,  # Nucleus sampling for diversity
                stream=False  # Non-streaming for simplicity
            )
            
            # Extract generated answer
            # response.choices[0] is the best completion
            # message.content is the actual text
            answer = response.choices[0].message.content
            
            # Calculate confidence score (simple heuristic)
            # Based on number and quality of retrieved chunks
            confidence = self._calculate_confidence(context_chunks, answer)
            
            # Prepare metadata about the generation
            metadata = {
                "model": self.model_name,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "context_chunks_count": len(context_chunks)
            }
            
            # Create QueryResponse object
            query_response = QueryResponse(
                answer=answer,
                sources=context_chunks,
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(
                f"Generated response: {len(answer)} chars, "
                f"confidence: {confidence:.2f}"
            )
            
            return query_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise RuntimeError(f"LLM generation failed: {str(e)}")
    
    
    def _build_prompt(
        self,
        query: str,
        context_chunks: List[RetrievedChunk]
    ) -> str:
        """
        Build the RAG prompt with context and query.
        
        Prompt engineering is crucial for good RAG performance.
        
        Structure:
        1. System instructions
        2. Context from retrieved chunks
        3. User query
        4. Instructions for response format
        
        Args:
            query: User's question
            context_chunks: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        # Start with system instructions
        prompt_parts = [
            "You are a helpful assistant that answers questions based on the provided context.",
            "Your task is to provide accurate, relevant, and concise answers.",
            "",
            "IMPORTANT INSTRUCTIONS:",
            "- Only use information from the provided context",
            "- If the context doesn't contain enough information, say so",
            "- Do NOT make up information",
            "- Cite sources when possible by mentioning the document name",
            "- Be concise but complete",
            "",
        ]
        
        # Add context if available
        if context_chunks:
            prompt_parts.append("CONTEXT:")
            prompt_parts.append("=" * 50)
            
            # Add each chunk with source information
            for i, retrieved_chunk in enumerate(context_chunks, start=1):
                chunk = retrieved_chunk.chunk
                score = retrieved_chunk.score
                
                # Format chunk with metadata
                chunk_text = (
                    f"\n[Source {i}] (Relevance: {score:.2f})\n"
                    f"Document: {chunk.metadata.filename}\n"
                    f"Content: {chunk.text}\n"
                    f"{'-' * 50}\n"
                )
                prompt_parts.append(chunk_text)
            
            prompt_parts.append("=" * 50)
            prompt_parts.append("")
        else:
            # No context available
            prompt_parts.append(
                "No relevant context was found in the knowledge base."
            )
            prompt_parts.append("")
        
        # Add the user's query
        prompt_parts.append("QUESTION:")
        prompt_parts.append(query)
        prompt_parts.append("")
        
        # Add response instructions
        prompt_parts.append("Please provide a comprehensive answer based on the context above.")
        
        # Join all parts
        prompt = "\n".join(prompt_parts)
        
        logger.debug(f"Built prompt: {len(prompt)} characters")
        
        return prompt
    
    
    def _prepare_messages(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for chat completion API.
        
        Chat models expect messages in a specific format:
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant response"},
            ...
        ]
        
        Args:
            prompt: Current prompt with query and context
            conversation_history: Previous conversation
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message
        # This sets the behavior and personality of the assistant
        system_message = {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant that helps users find "
                "information from their documents. You provide accurate, "
                "helpful, and well-sourced answers based on the given context."
            )
        }
        messages.append(system_message)
        
        # Add conversation history if provided
        # This enables multi-turn conversations
        if conversation_history:
            for message in conversation_history:
                # Validate message format
                if "role" in message and "content" in message:
                    messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
        
        # Add current prompt as user message
        user_message = {
            "role": "user",
            "content": prompt
        }
        messages.append(user_message)
        
        return messages
    
    
    def _calculate_confidence(
        self,
        context_chunks: List[RetrievedChunk],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        This is a heuristic based on:
        1. Number of context chunks
        2. Average similarity score
        3. Answer length (as proxy for completeness)
        
        Returns value between 0.0 and 1.0
        
        Args:
            context_chunks: Retrieved chunks used
            answer: Generated answer
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence starts at 0
        confidence = 0.0
        
        # Factor 1: Number of chunks (max 0.3)
        # More chunks = more context = higher confidence
        if context_chunks:
            # Normalize to 0-0.3 based on number of chunks
            chunk_score = min(len(context_chunks) / 5.0, 1.0) * 0.3
            confidence += chunk_score
        
        # Factor 2: Average similarity score (max 0.5)
        # Higher similarity = more relevant context
        if context_chunks:
            avg_score = sum(c.score for c in context_chunks) / len(context_chunks)
            similarity_score = avg_score * 0.5
            confidence += similarity_score
        
        # Factor 3: Answer completeness (max 0.2)
        # Longer answers tend to be more complete
        # (This is a rough heuristic)
        if len(answer) > 50:  # At least 50 characters
            completeness_score = min(len(answer) / 500.0, 1.0) * 0.2
            confidence += completeness_score
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    
    def generate_streaming_response(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Generate streaming response for real-time display.
        
        Streaming allows showing the response token-by-token
        as it's being generated, providing better UX.
        
        Args:
            query: User's question
            context_chunks: Retrieved context
            conversation_history: Previous messages
            
        Yields:
            Chunks of generated text
            
        Raises:
            RuntimeError: If streaming fails
        """
        logger.info("Starting streaming response generation...")
        
        # Build prompt and messages
        prompt = self._build_prompt(query, context_chunks)
        messages = self._prepare_messages(prompt, conversation_history)
        
        try:
            # Call Groq API with streaming enabled
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True  # Enable streaming
            )
            
            # Yield each chunk as it arrives
            for chunk in stream:
                # Extract delta (new content)
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
            
            logger.info("Streaming completed")
            
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise RuntimeError(f"Streaming generation failed: {str(e)}")
    
    
    
    def answer_with_citations(
        self,
        query: str,
        context_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """
        Generate answer with explicit inline citations.
        
        Citations help users verify information and build trust.
        Format: "Information [1]" where [1] refers to source 1.
        
        Args:
            query: User's question
            context_chunks: Retrieved context
            
        Returns:
            Dictionary with answer and citation mapping
        """
        logger.info("Generating answer with citations...")
        
        # Build prompt with citation instructions
        prompt_parts = [
            "Answer the following question using the provided sources.",
            "Include inline citations in the format [1], [2], etc.",
            "Each citation number corresponds to a source below.",
            "",
            "SOURCES:",
        ]
        
        # Add numbered sources
        for i, retrieved_chunk in enumerate(context_chunks, start=1):
            chunk = retrieved_chunk.chunk
            prompt_parts.append(
                f"\n[{i}] {chunk.metadata.filename}\n{chunk.text}\n"
            )
        
        prompt_parts.append(f"\nQUESTION: {query}")
        prompt_parts.append("\nProvide your answer with inline citations:")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that provides answers with "
                            "explicit citations. Always cite your sources using [1], [2], etc."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content
            
            # Create citation mapping
            # Maps citation numbers to source information
            citations = {}
            for i, retrieved_chunk in enumerate(context_chunks, start=1):
                chunk = retrieved_chunk.chunk
                citations[str(i)] = {
                    "document": chunk.metadata.filename,
                    "document_id": chunk.metadata.document_id,
                    "chunk_index": chunk.chunk_index,
                    "relevance_score": retrieved_chunk.score
                }
            
            return {
                "answer": answer,
                "citations": citations,
                "total_sources": len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Citation generation failed: {str(e)}")
            raise RuntimeError(f"Answer generation failed: {str(e)}")
    
    
    def check_answer_quality(
        self,
        query: str,
        answer: str,
        context_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated answer.
        
        Checks:
        1. Relevance to query
        2. Grounding in context
        3. Completeness
        4. Potential hallucinations
        
        Args:
            query: Original query
            answer: Generated answer
            context_chunks: Context used
            
        Returns:
            Quality assessment dictionary
        """
        logger.info("Checking answer quality...")
        
        # Build evaluation prompt
        eval_prompt = f"""Evaluate the quality of this answer:

                QUERY: {query}

                ANSWER: {answer}

                CONTEXT AVAILABLE:
                {len(context_chunks)} chunks with average relevance {sum(c.score for c in context_chunks) / len(context_chunks) if context_chunks else 0:.2f}

                Evaluate on a scale of 1-10:
                1. Relevance: Does it answer the query?
                2. Accuracy: Is it grounded in the context?
                3. Completeness: Is the answer complete?

                Provide scores in JSON format:
                {{"relevance": X, "accuracy": X, "completeness": X, "issues": ["issue1", ...]}}"""
        
        try:
            # Get evaluation from LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide objective assessments."
                    },
                    {
                        "role": "user",
                        "content": eval_prompt
                    }
                ],
                max_tokens=300,
                temperature=0.1  # Very low for factual evaluation
            )
            
            # Parse JSON response
            eval_text = response.choices[0].message.content
            
            # Try to extract JSON
            # Look for JSON between ```json and ```
            import re
            json_match = re.search(r'```json\n(.*?)\n```', eval_text, re.DOTALL)
            if json_match:
                eval_json = json.loads(json_match.group(1))
            else:
                # Try to parse entire response as JSON
                eval_json = json.loads(eval_text)
            
            return eval_json
            
        except Exception as e:
            logger.warning(f"Quality check failed: {str(e)}")
            # Return default assessment
            return {
                "relevance": 7,
                "accuracy": 7,
                "completeness": 7,
                "issues": ["Could not evaluate automatically"]
            }
    
    
    def rephrase_query(self, query: str) -> str:
        """
        Rephrase user query for better retrieval.
        
        Sometimes user queries are ambiguous or poorly worded.
        Rephrasing can improve retrieval quality.
        
        Args:
            query: Original user query
            
        Returns:
            Rephrased query
        """
        logger.info(f"Rephrasing query: {query}")
        
        prompt = f"""Rephrase the following query to be more clear and specific for document search.
                Keep it concise (1-2 sentences max).

                ORIGINAL QUERY: {query}

                REPHRASED QUERY:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You rephrase queries to be clear and searchable."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            rephrased = response.choices[0].message.content.strip()
            logger.info(f"Rephrased to: {rephrased}")
            
            return rephrased
            
        except Exception as e:
            logger.warning(f"Rephrasing failed: {str(e)}")
            # Return original on failure
            return query
    
# ========== Utility Functions ==========
def get_llm_service() -> LLMService:
    """
    Factory function to get LLM service instance.
    
    Returns:
        LLMService instance
    """
    return LLMService()


def format_response_for_display(response: QueryResponse) -> str:
    """
    Format QueryResponse for console display.
    
    Args:
        response: QueryResponse object
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 80,
        "ANSWER:",
        "-" * 80,
        response.answer,
        "",
        "SOURCES:",
        "-" * 80,
    ]
    
    # Add sources
    for i, retrieved_chunk in enumerate(response.sources, start=1):
        chunk = retrieved_chunk.chunk
        score = retrieved_chunk.score
        lines.append(
            f"[{i}] {chunk.metadata.filename} "
            f"(relevance: {score:.2f})"
        )
        lines.append(f"    {chunk.text[:100]}...")
        lines.append("")
    
    # Add metadata
    lines.append("METADATA:")
    lines.append("-" * 80)
    lines.append(f"Confidence: {response.confidence:.2f}")
    lines.append(f"Tokens used: {response.metadata.get('tokens_used', 'N/A')}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def calculate_response_cost(response: QueryResponse, cost_per_1k_tokens: float = 0.0001) -> float:
    """
    Calculate the cost of generating a response.
    
    Useful for:
    - Budget tracking
    - Usage analytics
    - Cost optimization
    
    Args:
        response: QueryResponse with metadata
        cost_per_1k_tokens: Cost per 1000 tokens
        
    Returns:
        Cost in dollars
    """
    tokens_used = response.metadata.get('tokens_used', 0)
    cost = (tokens_used / 1000) * cost_per_1k_tokens
    return cost