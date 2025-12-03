"""
Debug script to identify import issues
Run this to see what's wrong
"""

import os
import sys
import importlib

print("🔍 Debugging Import Issues")
print("="*50)

# Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# List all Python files in current directory
print("\n📁 Python files in current directory:")
py_files = [f for f in os.listdir('.') if f.endswith('.py')]
for f in py_files:
    size = os.path.getsize(f)
    print(f"  ✓ {f} ({size} bytes)")

# Try to import each module
print("\n📦 Testing imports:")

modules_to_test = [
    'database_manager',
    'arxiv_bot',
    'pdf_parser',
    'vector_store',
    'orchestrator'
]

for module_name in modules_to_test:
    print(f"\n{module_name}.py:")
    
    # Check if file exists
    if not os.path.exists(f"{module_name}.py"):
        print(f"  ✗ File not found!")
        continue
    
    # Try to import the module
    try:
        module = importlib.import_module(module_name)
        print(f"  ✓ Module imported successfully")
        
        # Check for expected classes
        if module_name == 'database_manager':
            if hasattr(module, 'DatabaseManager'):
                print(f"  ✓ DatabaseManager class found")
            else:
                print(f"  ✗ DatabaseManager class NOT found")
                
        elif module_name == 'arxiv_bot':
            if hasattr(module, 'ArxivBot'):
                print(f"  ✓ ArxivBot class found")
            else:
                print(f"  ✗ ArxivBot class NOT found")
                print(f"  Available: {dir(module)}")
                
        elif module_name == 'pdf_parser':
            if hasattr(module, 'PDFParser'):
                print(f"  ✓ PDFParser class found")
            else:
                print(f"  ✗ PDFParser class NOT found")
                
        elif module_name == 'vector_store':
            if hasattr(module, 'VectorStore'):
                print(f"  ✓ VectorStore class found")
            else:
                print(f"  ✗ VectorStore class NOT found")
                
        elif module_name == 'orchestrator':
            if hasattr(module, 'PipelineOrchestrator'):
                print(f"  ✓ PipelineOrchestrator class found")
            else:
                print(f"  ✗ PipelineOrchestrator class NOT found")
                
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
    except SyntaxError as e:
        print(f"  ✗ Syntax error in file: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")

# Test ArxivBot specifically
print("\n" + "="*50)
print("🔬 Detailed ArxivBot Test:")

try:
    # First check if the file exists and can be read
    with open('arxiv_bot.py', 'r') as f:
        content = f.read()
        print(f"  ✓ File readable ({len(content)} characters)")
        
        # Check if class definition exists
        if 'class ArxivBot' in content:
            print(f"  ✓ ArxivBot class definition found in file")
        else:
            print(f"  ✗ ArxivBot class definition NOT found in file")
            print("  First 500 characters of file:")
            print(content[:500])
            
except Exception as e:
    print(f"  ✗ Cannot read arxiv_bot.py: {e}")

# Try direct import
print("\n🧪 Direct import test:")
try:
    from arxiv_bot import ArxivBot
    print("  ✓ Direct import successful!")
    
    # Try to create instance
    bot = ArxivBot()
    print("  ✓ ArxivBot instance created!")
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
except Exception as e:
    print(f"  ✗ Error creating instance: {e}")
    
print("\n" + "="*50)
print("📝 Diagnosis complete!")
print("\nIf you see errors above, try:")
print("1. Make sure all files are saved in the same directory")
print("2. Check that no files are empty or corrupted")
print("3. Ensure you're running from the correct directory")
print("4. Try restarting your Python interpreter")