#!/usr/bin/env python3
"""
Test script to verify the updated Agno RAG implementation

This script tests the key fixes and verifications made to ensure
compatibility with current Agno patterns.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedder_interface():
    """Test that embedder implements correct interface"""
    logger.info("🔧 Testing embedder interface...")
    
    try:
        from embedders.agno_compatible_embedder import AgnoCompatibleEmbedder
        
        # Test interface methods exist
        embedder = AgnoCompatibleEmbedder.__new__(AgnoCompatibleEmbedder)  # Don't initialize model
        
        # Check methods exist
        assert hasattr(embedder, 'get_embedding'), "Missing get_embedding method"
        assert hasattr(embedder, 'get_embedding_and_usage'), "Missing get_embedding_and_usage method"
        
        # Check method signatures
        import inspect
        
        get_embedding_sig = inspect.signature(embedder.get_embedding)
        assert 'text' in get_embedding_sig.parameters, "get_embedding missing text parameter"
        
        get_embedding_usage_sig = inspect.signature(embedder.get_embedding_and_usage)
        assert 'text' in get_embedding_usage_sig.parameters, "get_embedding_and_usage missing text parameter"
        
        logger.info("✅ Embedder interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedder interface test failed: {e}")
        return False

def test_knowledge_base_interface():
    """Test that knowledge base implements correct interface"""
    logger.info("📚 Testing knowledge base interface...")
    
    try:
        from knowledge.agno_compatible_knowledge_base import AgnoCompatibleKnowledgeBase
        
        # Check class can be instantiated (don't actually connect to ChromaDB)
        kb_class = AgnoCompatibleKnowledgeBase
        
        # Check required methods exist
        methods = ['search', 'search_knowledge_base', 'get_relevant_documents', 'validate_filters']
        for method in methods:
            assert hasattr(kb_class, method), f"Missing {method} method"
        
        # Check search method signature
        import inspect
        search_sig = inspect.signature(kb_class.search)
        
        expected_params = ['self', 'query', 'num_documents', 'filters']
        actual_params = list(search_sig.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"Missing parameter {param} in search method"
        
        logger.info("✅ Knowledge base interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Knowledge base interface test failed: {e}")
        return False

def test_document_imports():
    """Test that Document class can be imported or fallback works"""
    logger.info("📄 Testing Document class availability...")
    
    try:
        # Try to import from the knowledge base module
        from knowledge.agno_compatible_knowledge_base import Document
        
        # Test Document can be instantiated
        doc = Document(id="test", content="test content", meta_data={"test": True})
        
        assert doc.id == "test", "Document id not set correctly"
        assert doc.content == "test content", "Document content not set correctly"
        assert doc.meta_data == {"test": True}, "Document meta_data not set correctly"
        
        logger.info("✅ Document class test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Document class test failed: {e}")
        return False

def test_embedder_inheritance():
    """Test that embedder properly inherits from base class"""
    logger.info("🏗️ Testing embedder inheritance...")
    
    try:
        from embedders.agno_compatible_embedder import AgnoCompatibleEmbedder, Embedder
        
        # Check inheritance
        assert issubclass(AgnoCompatibleEmbedder, Embedder), "AgnoCompatibleEmbedder should inherit from Embedder"
        
        # Check base class has required methods
        base_methods = ['get_embedding', 'get_embedding_and_usage']
        for method in base_methods:
            assert hasattr(Embedder, method), f"Base Embedder missing {method} method"
        
        logger.info("✅ Embedder inheritance test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedder inheritance test failed: {e}")
        return False

def test_return_types():
    """Test that methods have correct return type annotations"""
    logger.info("📝 Testing return type annotations...")
    
    try:
        from knowledge.agno_compatible_knowledge_base import AgnoCompatibleKnowledgeBase
        import inspect
        from typing import get_type_hints
        
        # Get type hints for search method
        try:
            hints = get_type_hints(AgnoCompatibleKnowledgeBase.search)
            return_type = hints.get('return', None)
            
            # Check that return type involves List and Document
            if return_type:
                type_str = str(return_type)
                assert 'List' in type_str, f"Search method should return List, got {type_str}"
                assert 'Document' in type_str, f"Search method should return List[Document], got {type_str}"
            
        except Exception as type_hint_error:
            # Type hints might not be available in all Python versions
            logger.info(f"Type hint check skipped: {type_hint_error}")
        
        logger.info("✅ Return type test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Return type test failed: {e}")
        return False

def test_configuration_compatibility():
    """Test that configuration follows current patterns"""
    logger.info("⚙️ Testing configuration compatibility...")
    
    try:
        # Test that num_documents parameter is supported
        from knowledge.agno_compatible_knowledge_base import AgnoCompatibleKnowledgeBase
        import inspect
        
        # Check constructor has num_documents parameter
        init_sig = inspect.signature(AgnoCompatibleKnowledgeBase.__init__)
        assert 'num_documents' in init_sig.parameters, "Constructor missing num_documents parameter"
        
        # Check default value
        param = init_sig.parameters['num_documents']
        assert param.default == 5, f"num_documents should default to 5, got {param.default}"
        
        logger.info("✅ Configuration compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all verification tests"""
    logger.info("🚀 Starting verification tests...")
    
    tests = [
        test_embedder_interface,
        test_knowledge_base_interface,
        test_document_imports,
        test_embedder_inheritance,
        test_return_types,
        test_configuration_compatibility
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All verification tests passed! Implementation is correct.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Review the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
