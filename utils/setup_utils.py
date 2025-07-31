"""
Utility functions for the Agno RAG solution.

This module provides helper functions for environment setup,
data loading, and system integration.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment(env_file: str = ".env") -> Dict[str, str]:
    """
    Setup environment variables and validate required keys.
    
    Args:
        env_file: Path to environment file
        
    Returns:
        Dictionary of environment variables
    """
    # Load environment file if it exists
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")
    else:
        logger.warning(f"Environment file {env_file} not found")
    
    # Required environment variables
    required_vars = [
        "OPENROUTER_API_KEY",
        "EMBEDDER_PROVIDER", 
        "HF_LOCAL_EMBEDDER"
    ]
    
    # Optional environment variables with defaults
    optional_vars = {
        "MODEL_PROVIDER": "openrouter",
        "MODEL_ID": "google/gemini-2.5-flash", 
        "TEMPERATURE": "0.1",
        "MAX_TOKENS": "3000",
        "CHROMA_DB_PATH": "database",
        "COLLECTION_NAME": "medical_coding_knowledge"
    }
    
    env_config = {}
    missing_vars = []
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if value:
            env_config[var] = value
        else:
            missing_vars.append(var)
    
    # Set optional variables with defaults
    for var, default in optional_vars.items():
        env_config[var] = os.getenv(var, default)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("✅ Environment setup completed")
    return env_config


def setup_project_paths() -> Dict[str, Path]:
    """
    Setup and validate project directory paths.
    
    Returns:
        Dictionary of project paths
    """
    # Get current directory (should be the solution directory)
    current_dir = Path.cwd()
    
    # Define project structure
    paths = {
        "root": current_dir,
        "embedders": current_dir / "embedders",
        "knowledge": current_dir / "knowledge", 
        "agents": current_dir / "agents",
        "utils": current_dir / "utils",
        "data": current_dir / "data",
        "database": current_dir / "database",
        "docs": current_dir.parent / "docs"
    }
    
    # Create missing directories
    for name, path in paths.items():
        if name in ["data", "database"]:  # Create these if they don't exist
            path.mkdir(exist_ok=True)
            logger.info(f"Ensured directory exists: {path}")
    
    # Validate critical directories exist
    critical_dirs = ["embedders", "knowledge", "agents", "utils"]
    for dir_name in critical_dirs:
        if not paths[dir_name].exists():
            raise FileNotFoundError(f"Critical directory missing: {paths[dir_name]}")
    
    logger.info("✅ Project paths setup completed")
    return paths


def add_solution_to_path():
    """Add solution directories to Python path for imports."""
    paths = setup_project_paths()
    
    # Add solution root to path
    root_path = str(paths["root"])
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
        logger.info(f"Added to Python path: {root_path}")
    
    # Add subdirectories to path
    for subdir in ["embedders", "knowledge", "agents", "utils"]:
        subdir_path = str(paths[subdir])
        if subdir_path not in sys.path:
            sys.path.insert(0, subdir_path)
            logger.info(f"Added to Python path: {subdir_path}")


def validate_dependencies() -> Dict[str, bool]:
    """
    Validate that required dependencies are available.
    
    Returns:
        Dictionary of dependency availability
    """
    dependencies = {
        "torch": False,
        "transformers": False,
        "chromadb": False,
        "agno": False,
        "dotenv": False,
        "numpy": False
    }
    
    for dep in dependencies:
        try:
            __import__(dep.replace("dotenv", "python_dotenv"))
            dependencies[dep] = True
        except ImportError:
            logger.warning(f"Dependency not available: {dep}")
    
    missing = [dep for dep, available in dependencies.items() if not available]
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.info("Install missing dependencies with: pip install torch transformers chromadb agno python-dotenv numpy")
    else:
        logger.info("✅ All dependencies available")
    
    return dependencies


def find_icd_data_file(search_paths: List[str] = None) -> Optional[str]:
    """
    Find ICD-10 data file in common locations.
    
    Args:
        search_paths: Additional paths to search
        
    Returns:
        Path to ICD data file if found
    """
    if search_paths is None:
        search_paths = []
    
    # Common file patterns and locations
    file_patterns = [
        "icd10cm_codes_2026.txt",
        "icd10cm_codes*.txt", 
        "icd_codes.txt",
        "medical_codes.txt"
    ]
    
    search_locations = [
        ".",
        "./data",
        "../data", 
        "./database",
        "../database",
        "/kaggle/input/icd10cm-codes-2026-txt/"
    ] + search_paths
    
    for location in search_locations:
        location_path = Path(location)
        if not location_path.exists():
            continue
            
        for pattern in file_patterns:
            if "*" in pattern:
                # Use glob for wildcards
                matches = list(location_path.glob(pattern))
                if matches:
                    found_file = str(matches[0])
                    logger.info(f"Found ICD data file: {found_file}")
                    return found_file
            else:
                # Direct file check
                file_path = location_path / pattern
                if file_path.exists():
                    found_file = str(file_path)
                    logger.info(f"Found ICD data file: {found_file}")
                    return found_file
    
    logger.warning("ICD data file not found in common locations")
    return None


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        System information dictionary
    """
    import platform
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        cuda_available = False
        cuda_device_count = 0
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "working_directory": str(Path.cwd()),
        "python_path": sys.path[:3]  # First 3 entries
    }


def create_sample_environment_file(output_path: str = ".env.sample"):
    """
    Create a sample environment file with required variables.
    
    Args:
        output_path: Path to output the sample file
    """
    sample_content = """# Agno RAG Solution Environment Variables

# Required API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
# Alternative: OPENAI_API_KEY=your_openai_api_key_here

# Embedder Configuration
EMBEDDER_PROVIDER=huggingface-local
HF_LOCAL_EMBEDDER=Qwen/Qwen2.5-Coder-0.5B

# Model Configuration  
MODEL_PROVIDER=openrouter
MODEL_ID=google/gemini-2.5-flash
TEMPERATURE=0.1
MAX_TOKENS=3000

# Database Configuration
CHROMA_DB_PATH=database
COLLECTION_NAME=medical_coding_knowledge

# Optional: HuggingFace Token for private models
# HF_TOKEN=your_huggingface_token_here

# Optional: Additional API Keys
# GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
"""
    
    with open(output_path, 'w') as f:
        f.write(sample_content)
    
    logger.info(f"Sample environment file created: {output_path}")


def print_setup_summary(env_config: Dict[str, str], paths: Dict[str, Path]):
    """
    Print a summary of the setup configuration.
    
    Args:
        env_config: Environment configuration
        paths: Project paths
    """
    print("\n" + "="*60)
    print("AGNO RAG SOLUTION SETUP SUMMARY")
    print("="*60)
    
    print(f"\n📁 Project Structure:")
    for name, path in paths.items():
        status = "✅" if path.exists() else "❌"
        print(f"   {status} {name}: {path}")
    
    print(f"\n🔧 Environment Configuration:")
    # Show safe environment variables (not API keys)
    safe_vars = ["EMBEDDER_PROVIDER", "MODEL_PROVIDER", "MODEL_ID", "TEMPERATURE", "MAX_TOKENS"]
    for var in safe_vars:
        if var in env_config:
            print(f"   {var}: {env_config[var]}")
    
    # Show API key status without values
    api_keys = ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"]
    print(f"\n🔑 API Key Status:")
    for key in api_keys:
        status = "✅ Set" if env_config.get(key) else "❌ Not set"
        print(f"   {key}: {status}")
    
    print(f"\n🏗️  Next Steps:")
    print("   1. Run the notebook cells to test the implementation")
    print("   2. Load your ICD-10 data into the knowledge base")
    print("   3. Test the agent with medical coding queries")
    print("   4. Monitor search functionality and adjust as needed")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test utility functions
    print("Testing Agno RAG Solution Utilities...")
    
    # Test system info
    sys_info = get_system_info()
    print(f"System: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"CUDA: {sys_info['cuda_available']}")
    
    # Test dependency validation
    deps = validate_dependencies()
    available_count = sum(deps.values())
    print(f"Dependencies: {available_count}/{len(deps)} available")
    
    print("✅ Utilities test completed!")
