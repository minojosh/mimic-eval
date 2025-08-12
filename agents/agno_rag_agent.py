"""
Agno-Compatible Agent Implementation

This module provides agent setup and configuration that follows Agno's
patterns for medical coding RAG systems. Based on research findings
from Agno's framework documentation and examples.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgnoRagAgent:
    """
    Agno-compatible RAG agent for medical coding tasks.
    
    This class encapsulates the agent creation and configuration
    following Agno's best practices for knowledge-based agents.
    """
    
    def __init__(
        self,
        knowledge_base,
        model_provider: str = "openrouter",
        model_id: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        max_tokens: int = 6000
    ):
        """
        Initialize the Agno RAG agent.
        
        Args:
            knowledge_base: Agno-compatible knowledge base instance
            model_provider: Model provider ('openrouter', 'openai', etc.)
            model_id: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.knowledge_base = knowledge_base
        self.model_provider = model_provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.agent = None
        self.model = None
        
        self._initialize_model()
        self._create_agent()
    
    def _initialize_model(self):
        """Initialize the language model based on provider."""
        try:
            if self.model_provider == "openrouter":
                from agno.models.openrouter import OpenRouter
                
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
                
                self.model = OpenRouter(
                    id=self.model_id,
                    api_key=api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
            elif self.model_provider == "openai":
                from agno.models.openai import OpenAIChat
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                
                self.model = OpenAIChat(
                    id=self.model_id,
                    api_key=api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
            
            logger.info(f"✅ Model initialized: {self.model_provider}/{self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _create_agent(self):
        """Create the Agno agent with proper configuration."""
        try:
            from agno.agent import Agent
            
            # Medical coding specific instructions
            instructions = [
                "You are a medical coding expert specializing in ICD-10 codes.",
                "ALWAYS search the knowledge base for relevant ICD codes before responding.",
                "Use the search_knowledge_base function to find relevant medical conditions.",
                "When analyzing medical conditions, search for related ICD codes first.",
                "Provide exact ICD codes found in the knowledge base along with their descriptions.",
                "Format ICD codes clearly with the code and description.",
                "If multiple codes are relevant, list them all with explanations.",
                "Be thorough in your search - try different related terms if initial search yields few results.",
                "Always explain your reasoning for code selection."
            ]
            
            self.agent = Agent(
                knowledge=self.knowledge_base,
                search_knowledge=True,  # Enable automatic knowledge search
                model=self.model,
                instructions=instructions,
                markdown=True,  # Enable markdown formatting
                show_tool_calls=True,  # Show knowledge base searches
                debug_mode=False  # Set to True for debugging
            )
            
            logger.info("✅ Agno agent created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    def query(self, question: str, stream: bool = False) -> str:
        """
        Query the agent with a medical coding question.
        
        Args:
            question: Medical coding question
            stream: Whether to stream the response
            
        Returns:
            Agent's response
        """
        try:
            if stream:
                # For streaming, print directly
                self.agent.print_response(question, stream=True)
                return "Response streamed above"
            else:
                # For non-streaming, return the response
                response = self.agent.run(question)
                return response.content if hasattr(response, 'content') else str(response)
                
        except Exception as e:
            logger.error(f"Error querying agent: {e}")
            return f"Error: {str(e)}"
    
    def test_knowledge_search(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Test the agent's knowledge search functionality.
        
        Args:
            test_queries: List of test queries, uses defaults if None
            
        Returns:
            Test results
        """
        if test_queries is None:
            test_queries = [
                "COVID-19 ICD codes",
                "diabetes mellitus",
                "respiratory failure",
                "hypertension",
                "pneumonia"
            ]
        
        test_results = {
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "query_results": []
        }
        
        logger.info(f"Testing agent knowledge search with {len(test_queries)} queries...")
        
        for query in test_queries:
            try:
                logger.info(f"Testing query: '{query}'")
                
                # Test direct knowledge base search first
                kb_results = self.knowledge_base.search(query, limit=5)
                
                # Test agent query
                agent_response = self.query(f"Find ICD-10 codes for: {query}")
                
                query_result = {
                    "query": query,
                    "kb_results_count": len(kb_results),
                    "agent_response_length": len(agent_response),
                    "success": len(kb_results) > 0 and len(agent_response) > 50,
                    "sample_kb_results": [
                        {
                            "content": r.get("content", "")[:100] + "...",
                            "score": r.get("score", 0.0)
                        } for r in kb_results[:2]
                    ]
                }
                
                test_results["query_results"].append(query_result)
                
                if query_result["success"]:
                    test_results["successful_queries"] += 1
                else:
                    test_results["failed_queries"] += 1
                    
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                test_results["failed_queries"] += 1
                test_results["query_results"].append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        success_rate = test_results["successful_queries"] / test_results["total_queries"]
        test_results["success_rate"] = success_rate
        
        logger.info(f"Knowledge search test completed: {success_rate:.2%} success rate")
        
        return test_results
    
    def analyze_discharge_note(self, note_text: str, stream: bool = True) -> str:
        """
        Analyze a discharge note and assign ICD-10 codes.
        
        Args:
            note_text: Discharge note text
            stream: Whether to stream the response
            
        Returns:
            Analysis with ICD-10 codes
        """
        prompt = f"""
        Analyze this discharge note and assign appropriate ICD-10-CM codes.
        
        Use multi-step reasoning:
        1. Identify all medical conditions mentioned in the note
        2. Search the knowledge base for relevant ICD-10-CM codes for each condition
        3. Assign specific codes with confidence scores
        4. Provide rationale for each code assignment
        
        DISCHARGE NOTE:
        {note_text}
        
        Please provide:
        - Primary diagnosis codes
        - Secondary diagnosis codes  
        - Procedure codes if applicable
        - Brief justification for each code
        """
        
        return self.query(prompt, stream=stream)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent configuration.
        
        Returns:
            Agent configuration info
        """
        return {
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "knowledge_base_info": self.knowledge_base.get_collection_info(),
            "has_agent": self.agent is not None,
            "has_model": self.model is not None
        }


def create_medical_coding_agent(
    knowledge_base,
    model_provider: str = "openrouter",
    model_id: str = "openai/gpt-oss-120b"
) -> AgnoRagAgent:
    """
    Factory function to create a medical coding RAG agent.
    
    Args:
        knowledge_base: Agno-compatible knowledge base
        model_provider: Model provider
        model_id: Model identifier
        
    Returns:
        Configured RAG agent
    """
    return AgnoRagAgent(
        knowledge_base=knowledge_base,
        model_provider=model_provider,
        model_id=model_id
    )


def create_agent_with_environment_config(knowledge_base) -> AgnoRagAgent:
    """
    Create agent using environment variables for configuration.
    
    Args:
        knowledge_base: Agno-compatible knowledge base
        
    Returns:
        Configured RAG agent
    """
    # Get configuration from environment
    model_provider = os.getenv("MODEL_PROVIDER", "openrouter")
    model_id = os.getenv("MODEL_ID", "openai/gpt-oss-120b")
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("MAX_TOKENS", "6000"))
    
    return AgnoRagAgent(
        knowledge_base=knowledge_base,
        model_provider=model_provider,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens
    )


if __name__ == "__main__":
    # Test agent creation (requires knowledge base)
    print("Testing Agno-Compatible Agent...")
    print("✅ Agent module loaded successfully!")
    print("Note: Full testing requires knowledge base and model API keys")
