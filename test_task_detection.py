#!/usr/bin/env python3
"""
Test script to verify OpenEnv task and grader detection
Run this locally before submitting to Hugging Face
"""

import os
import sys
import importlib
import yaml

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 OpenEnv Task & Grader Detection Test")
print("=" * 60)

# ===== TEST 1: Check openenv.yaml =====
print("\n📋 TEST 1: Parsing openenv.yaml...")
try:
    with open('openenv.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tasks = config.get('tasks', [])
    print(f"✅ Found {len(tasks)} task(s) defined in openenv.yaml")
    
    for i, task in enumerate(tasks, 1):
        print(f"\n   Task {i}:")
        print(f"     - id: {task.get('id')}")
        print(f"     - name: {task.get('name')}")
        print(f"     - grader: {task.get('grader')}")
        print(f"     - grader_module: {task.get('grader_module', 'NOT SPECIFIED')}")
        
    if len(tasks) != 3:
        print(f"\n⚠️ WARNING: Expected 3 tasks, but found {len(tasks)}")
        
except FileNotFoundError:
    print("❌ openenv.yaml not found!")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error parsing openenv.yaml: {e}")
    sys.exit(1)

# ===== TEST 2: Try to import graders =====
print("\n" + "=" * 60)
print("📋 TEST 2: Attempting to import graders...")

grader_classes = ['BugDetectionGrader', 'BugClassificationGrader', 'FixSuggestionGrader']
imported_graders = []

for grader_name in grader_classes:
    try:
        # Try different import paths
        import_paths = [
            f"backend.environment.tasks.{grader_name}",
            f"backend.tasks.{grader_name}",
            f"tasks.{grader_name}",
            f"environment.tasks.{grader_name}",
        ]
        
        found = False
        for import_path in import_paths:
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                grader = getattr(module, class_name)
                imported_graders.append(grader_name)
                print(f"✅ {grader_name} -> {import_path}")
                found = True
                break
            except (ImportError, AttributeError):
                continue
        
        if not found:
            print(f"❌ {grader_name} -> NOT FOUND in any import path")
            
    except Exception as e:
        print(f"❌ {grader_name} -> Error: {e}")

print(f"\n📊 Successfully imported: {len(imported_graders)}/3 graders")

# ===== TEST 3: Check environment module =====
print("\n" + "=" * 60)
print("📋 TEST 3: Checking environment module...")

try:
    # Try different environment import paths
    env_paths = [
        "backend.environment",
        "backend.environment.environment",
        "environment",
    ]
    
    for env_path in env_paths:
        try:
            env_module = importlib.import_module(env_path)
            if hasattr(env_module, 'CodeReviewEnvironment'):
                print(f"✅ CodeReviewEnvironment found at: {env_path}")
                break
        except ImportError:
            continue
    else:
        print("❌ CodeReviewEnvironment not found in any import path")
        
except Exception as e:
    print(f"❌ Error checking environment: {e}")

# ===== TEST 4: Create environment and check tasks =====
print("\n" + "=" * 60)
print("📋 TEST 4: Creating environment and checking runtime tasks...")

try:
    # Try to import and create environment
    from backend.environment import CodeReviewEnvironment
    env = CodeReviewEnvironment(use_dynamic_snippets=False)
    
    # Check task configs
    print("\n✅ Environment created successfully!")
    print("\nRuntime task detection:")
    
    for task_id in [1, 2, 3]:
        env.current_task = task_id
        config = env._get_task_config()
        grader = config.get('grader')
        grader_name = grader.__class__.__name__ if grader else "None"
        print(f"\n   Task {task_id}:")
        print(f"     - Name: {config.get('name')}")
        print(f"     - Grader: {grader_name}")
        print(f"     - Max steps: {config.get('max_steps')}")
        print(f"     - Difficulty: {config.get('difficulty')}")
    
    print("\n✅ All 3 tasks are properly configured at runtime!")
    
except ImportError as e:
    print(f"❌ Cannot import environment: {e}")
    print("\n   Try fixing imports in your files:")
    print("   - Make sure backend/__init__.py exists")
    print("   - Make sure backend/environment/__init__.py exists")
    print("   - Check that all imports use correct paths")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ===== TEST 5: Check file structure =====
print("\n" + "=" * 60)
print("📋 TEST 5: Checking file structure...")

import os

required_files = [
    'openenv.yaml',
    'Dockerfile',
    'inference.py',
    'backend/__init__.py',
    'backend/environment.py',
    'backend/models.py',
    'backend/tasks.py',
    'backend/app.py',
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - MISSING")

# ===== SUMMARY =====
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

if len(tasks) == 3 and len(imported_graders) == 3:
    print("\n✅✅✅ ALL CHECKS PASSED! ✅✅✅")
    print("\nYour environment has 3 tasks with proper graders.")
    print("OpenEnv Phase 2 should pass when you submit.")
else:
    print("\n❌❌❌ ISSUES DETECTED ❌❌❌")
    print(f"\n- Tasks in openenv.yaml: {len(tasks)} (should be 3)")
    print(f"- Graders imported: {len(imported_graders)} (should be 3)")
    print("\nFixes needed:")
    
    if len(tasks) != 3:
        print("  1. Add the missing task(s) to openenv.yaml")
    
    if len(imported_graders) != 3:
        print("  2. Fix grader imports:")
        print("     - Add __init__.py files in backend/")
        print("     - Ensure grader_module paths are correct in openenv.yaml")
        print("     - Check that grader classes are properly exported")
    
    print("\n  3. Run: python test_task_detection.py again after fixes")

print("\n" + "=" * 60)