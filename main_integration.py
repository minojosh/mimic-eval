"""
Main integration module for Agno RAG Solution

This module provides the main entry point for running the complete
Agno-compatible RAG system with local HuggingFace embeddings.

VERIFICATION: Updated to match current Agno v1.7.6 patterns exactly.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import solution components
try:
    from utils.setup_utils import (
        setup_environment, 
        setup_project_paths, 
        add_solution_to_path,
        validate_dependencies,
        find_icd_data_file,
        get_system_info,
        print_setup_summary
    )
    from embedders.agno_compatible_embedder import create_embedder
    from knowledge.agno_compatible_knowledge_base import create_knowledge_base
    from agents.agno_rag_agent import create_medical_coding_agent
    
    logger.info("✅ All solution components imported successfully")
except ImportError as e:
    logger.error(f"Failed to import solution components: {e}")
    logger.info("Make sure you're running from the agno_rag_solution directory")
    logger.info("Note: This is normal if Agno is not installed - fallback classes will be used")
    raise


class AgnoRagSystem:
    """
    Complete Agno RAG System for Medical Coding
    
    This class orchestrates all components of the RAG system:
    - Local HuggingFace embedder
    - ChromaDB knowledge base
    - Agno-compatible agent
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the complete RAG system.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = None
        self.paths = None
        self.embedder = None
        self.knowledge_base = None
        self.agent = None
        
        # Setup system
        self._setup_system(config_override)
    
    def _setup_system(self, config_override: Optional[Dict[str, Any]] = None):
        """Setup the complete system with all components."""
        logger.info("🚀 Initializing Agno RAG System...")
        
        try:
            # 1. Setup paths and environment
            add_solution_to_path()
            self.paths = setup_project_paths()
            self.config = setup_environment()
            
            # Apply any configuration overrides
            if config_override:
                self.config.update(config_override)
            
            # 2. Validate dependencies
            deps = validate_dependencies()
            missing_deps = [dep for dep, available in deps.items() if not available]
            if missing_deps:
                logger.warning(f"Missing dependencies: {missing_deps}")
                logger.info("Some features may not work without all dependencies")
            
            # 3. Print setup summary
            print_setup_summary(self.config, self.paths)
            
            logger.info("✅ System setup completed")
            
        except Exception as e:
            logger.error(f"System setup failed: {e}")
            raise
    
    def initialize_embedder(self, force_recreate: bool = False) -> bool:
        """
        Initialize the local HuggingFace embedder.
        
        Args:
            force_recreate: Force recreation even if already exists
            
        Returns:
            True if successful
        """
        if self.embedder and not force_recreate:
            logger.info("Embedder already initialized")
            return True
        
        try:
            logger.info("🔧 Initializing embedder...")
            
            model_id = self.config.get("HF_LOCAL_EMBEDDER", "Qwen/Qwen2.5-Coder-0.5B")
            self.embedder = create_embedder(model_id=model_id)
            
            # Test the embedder
            test_embedding = self.embedder.get_embedding("test text")
            logger.info(f"✅ Embedder initialized - embedding dimension: {len(test_embedding)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            return False
    
    def initialize_knowledge_base(self, force_recreate: bool = False) -> bool:
        """
        Initialize the ChromaDB knowledge base.
        
        Args:
            force_recreate: Force recreation even if already exists
            
        Returns:
            True if successful
        """
        if self.knowledge_base and not force_recreate:
            logger.info("Knowledge base already initialized")
            return True
        
        if not self.embedder:
            logger.error("Embedder must be initialized first")
            return False
        
        try:
            logger.info("📚 Initializing knowledge base...")
            
            chroma_db_path = self.config.get("CHROMA_DB_PATH", "database")
            collection_name = self.config.get("COLLECTION_NAME", "medical_coding_knowledge")
            
            self.knowledge_base = create_knowledge_base(
                chroma_db_path=chroma_db_path,
                collection_name=collection_name,
                embedder=self.embedder,
                num_documents=5
            )
            
            # Get collection info
            info = self.knowledge_base.get_collection_info()
            logger.info(f"✅ Knowledge base initialized - {info['document_count']} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    def load_icd_data(self, file_path: Optional[str] = None, recreate: bool = False) -> bool:
        """
        Load ICD-10 data into the knowledge base.
        
        Args:
            file_path: Path to ICD data file (auto-detect if None)
            recreate: Whether to recreate the collection
            
        Returns:
            True if successful
        """
        if not self.knowledge_base:
            logger.error("Knowledge base must be initialized first")
            return False
        
        try:
            # Find ICD data file if not provided
            if file_path is None:
                file_path = find_icd_data_file()
                if not file_path:
                    logger.error("No ICD data file found")
                    return False
            
            if not os.path.exists(file_path):
                logger.error(f"ICD data file not found: {file_path}")
                return False
            
            logger.info(f"📄 Loading ICD data from: {file_path}")
            
            # Load documents into knowledge base
            self.knowledge_base.load_documents_from_file(
                file_path=file_path,
                chunk_size=1000,
                overlap=100,
                recreate=recreate
            )
            
            # Verify loading
            info = self.knowledge_base.get_collection_info()
            logger.info(f"✅ ICD data loaded - {info['document_count']} documents total")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ICD data: {e}")
            return False
    
    def initialize_agent(self, force_recreate: bool = False) -> bool:
        """
        Initialize the Agno RAG agent.
        
        Args:
            force_recreate: Force recreation even if already exists
            
        Returns:
            True if successful
        """
        if self.agent and not force_recreate:
            logger.info("Agent already initialized")
            return True
        
        if not self.knowledge_base:
            logger.error("Knowledge base must be initialized first")
            return False
        
        try:
            logger.info("🤖 Initializing agent...")
            
            model_provider = self.config.get("MODEL_PROVIDER", "openrouter")
            model_id = self.config.get("MODEL_ID", "google/gemini-2.5-flash")
            
            self.agent = create_medical_coding_agent(
                knowledge_base=self.knowledge_base,
                model_provider=model_provider,
                model_id=model_id
            )
            
            logger.info(f"✅ Agent initialized with {model_provider}/{model_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return False
    
    def run_system_test(self) -> Dict[str, Any]:
        """
        Run a comprehensive system test.
        
        Returns:
            Test results
        """
        logger.info("🧪 Running system test...")
        
        test_results = {
            "embedder_test": False,
            "knowledge_base_test": False,
            "agent_test": False,
            "overall_success": False
        }
        
        try:
            # Test embedder
            if self.embedder:
                test_embedding = self.embedder.get_embedding("test medical condition")
                test_results["embedder_test"] = len(test_embedding) > 0
            
            # Test knowledge base
            if self.knowledge_base:
                kb_test = self.knowledge_base.test_search("diabetes")
                test_results["knowledge_base_test"] = kb_test.get("results_count", 0) > 0
            
            # Test agent
            if self.agent:
                agent_test = self.agent.test_knowledge_search(["COVID-19"])
                test_results["agent_test"] = agent_test.get("success_rate", 0) > 0
            
            # Overall success
            test_results["overall_success"] = all([
                test_results["embedder_test"],
                test_results["knowledge_base_test"], 
                test_results["agent_test"]
            ])
            
            logger.info(f"✅ System test completed - Success: {test_results['overall_success']}")
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    def query_agent(self, question: str, stream: bool = True) -> str:
        """
        Query the agent with a medical coding question.
        
        Args:
            question: Medical coding question
            stream: Whether to stream the response
            
        Returns:
            Agent response
        """
        if not self.agent:
            return "Error: Agent not initialized"
        
        return self.agent.query(question, stream=stream)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            System status information
        """
        return {
            "embedder_ready": self.embedder is not None,
            "knowledge_base_ready": self.knowledge_base is not None,
            "agent_ready": self.agent is not None,
            "config": self.config,
            "system_info": get_system_info(),
            "knowledge_base_info": self.knowledge_base.get_collection_info() if self.knowledge_base else {},
            "agent_info": self.agent.get_agent_info() if self.agent else {}
        }


def create_complete_system(config_override: Optional[Dict[str, Any]] = None) -> AgnoRagSystem:
    """
    Factory function to create a complete Agno RAG system.
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Initialized RAG system
    """
    return AgnoRagSystem(config_override=config_override)


def quick_setup_and_test() -> AgnoRagSystem:
    """
    Quick setup function for testing the complete system.
    
    Returns:
        Initialized and tested RAG system
    """
    logger.info("🚀 Quick setup and test...")
    
    # Create system
    system = create_complete_system()
    
    # Initialize all components
    if system.initialize_embedder():
        if system.initialize_knowledge_base():
            if system.initialize_agent():
                # Run system test
                test_results = system.run_system_test()
                if test_results["overall_success"]:
                    logger.info("✅ Quick setup completed successfully!")
                else:
                    logger.warning("⚠️ Quick setup completed with some failures")
            else:
                logger.error("❌ Agent initialization failed")
        else:
            logger.error("❌ Knowledge base initialization failed")
    else:
        logger.error("❌ Embedder initialization failed")
    
    return system


if __name__ == "__main__":
    # Run quick setup and test
    system = quick_setup_and_test()
    
    # Print final status
    status = system.get_system_status()
    print(f"\n📊 Final Status:")
    print(f"   Embedder: {'✅' if status['embedder_ready'] else '❌'}")
    print(f"   Knowledge Base: {'✅' if status['knowledge_base_ready'] else '❌'}")
    print(f"   Agent: {'✅' if status['agent_ready'] else '❌'}")
    
    if all([status['embedder_ready'], status['knowledge_base_ready'], status['agent_ready']]):
        print("\n🎉 System ready for medical coding queries!")
    else:
        print("\n⚠️ System partially ready - check logs for issues")
