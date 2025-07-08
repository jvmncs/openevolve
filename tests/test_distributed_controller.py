#!/usr/bin/env python
"""
Test script for the DistributedController

This script tests the basic functionality of the distributed architecture
without requiring a full Modal deployment.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Simple test program
TEST_PROGRAM = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

# Simple evaluation function
TEST_EVALUATOR = """
def evaluate(code, program_id):
    '''Simple evaluation that returns a score based on code length'''
    try:
        # Execute the code to check if it runs
        exec(code)
        # Score inversely proportional to code length (shorter is better)
        score = 1.0 / (len(code) + 1)
        return {"score": score, "length": len(code)}
    except Exception as e:
        return {"score": 0.0, "error": str(e)}
"""

async def test_distributed_controller():
    """Test the DistributedController without Modal deployment"""
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write test files
        program_file = temp_path / "test_program.py"
        program_file.write_text(TEST_PROGRAM)
        
        eval_file = temp_path / "test_evaluator.py" 
        eval_file.write_text(TEST_EVALUATOR)
        
        # Create minimal config
        config_content = """
max_iterations: 5
log_level: "INFO"

llm:
  models:
    - name: "test-model"
      weight: 1.0
  evaluator_models:
    - name: "test-model"
      weight: 1.0
  api_base: "http://localhost:8000"
  temperature: 0.7
  max_tokens: 1000

prompt:
  system_message: "You are a helpful coding assistant"

database:
  population_size: 10
  
evaluator:
  timeout: 60
  parallel_evaluations: 2
"""
        config_file = temp_path / "config.yaml"
        config_file.write_text(config_content)
        
        print("Testing DistributedController initialization...")
        
        try:
            # Test import
            from openevolve.distributed_controller import DistributedController
            print("✓ Successfully imported DistributedController")
            
            # Test initialization
            controller = DistributedController(
                initial_program_path=str(program_file),
                evaluation_file=str(eval_file),
                config_path=str(config_file),
                output_dir=str(temp_path / "output")
            )
            print("✓ Successfully initialized DistributedController")
            
            # Test configuration loading
            assert controller.config is not None
            assert controller.config.max_iterations == 5
            print("✓ Configuration loaded correctly")
            
            # Test initial program loading
            assert controller.initial_program_code == TEST_PROGRAM
            print("✓ Initial program loaded correctly")
            
            # Test Modal app reference
            assert controller.modal_app is not None
            print("✓ Modal app reference set")
            
            print("\n🎉 All basic tests passed!")
            print("\nNote: Full functionality requires Modal deployment.")
            print("To run with Modal: modal run test_distributed_controller.py")
            
        except ImportError as e:
            if "modal" in str(e):
                print("❌ Modal not available. Install with: pip install modal")
            else:
                print(f"❌ Import error: {e}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

def test_cli_integration():
    """Test CLI integration"""
    print("\nTesting CLI integration...")
    
    try:
        from openevolve.cli import parse_args
        
        # Test distributed flag parsing
        import sys
        original_argv = sys.argv
        
        try:
            # Test distributed flag
            sys.argv = ["cli.py", "program.py", "eval.py", "--distributed"]
            args = parse_args()
            assert args.distributed == True
            print("✓ CLI distributed flag works")
            
            # Test normal mode
            sys.argv = ["cli.py", "program.py", "eval.py"]
            args = parse_args()
            assert args.distributed == False
            print("✓ CLI normal mode works")
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")

if __name__ == "__main__":
    print("OpenEvolve Distributed Controller Test")
    print("=" * 40)
    
    # Test CLI first
    test_cli_integration()
    
    # Test controller
    asyncio.run(test_distributed_controller())
    
    print("\nTo run with actual Modal deployment:")
    print("  1. Install modal: pip install modal")
    print("  2. Authenticate: modal setup")
    print("  3. Run: python openevolve-run.py program.py eval.py --distributed")
