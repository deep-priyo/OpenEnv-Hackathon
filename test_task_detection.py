#!/usr/bin/env python3
"""
Strict OpenEnv Task & Grader Detection Test
Matches actual OpenEnv evaluation behavior
"""

import os
import sys
import importlib
import yaml

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 OpenEnv STRICT Task & Grader Detection Test")
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
        print(f"     - grader_module: {task.get('grader_module')}")

    if len(tasks) < 3:
        print(f"\n❌ ERROR: At least 3 tasks required (found {len(tasks)})")

except Exception as e:
    print(f"❌ Failed to parse openenv.yaml: {e}")
    sys.exit(1)

# ===== TEST 2: STRICT grader import =====
print("\n" + "=" * 60)
print("📋 TEST 2: Strict grader import (OpenEnv-compatible)...")

grader_module_path = "backend.environment.tasks"
grader_classes = ['BugDetectionGrader', 'BugClassificationGrader', 'FixSuggestionGrader']
imported_graders = []

try:
    module = importlib.import_module(grader_module_path)
    print(f"✅ Module imported: {grader_module_path}")

    for grader_name in grader_classes:
        if hasattr(module, grader_name):
            print(f"✅ {grader_name} found")
            imported_graders.append(grader_name)
        else:
            print(f"❌ {grader_name} NOT found")

except ImportError as e:
    print(f"❌ FAILED to import module '{grader_module_path}': {e}")

print(f"\n📊 Successfully imported: {len(imported_graders)}/3 graders")

# ===== TEST 3: Environment import =====
print("\n" + "=" * 60)
print("📋 TEST 3: Checking environment module...")

try:
    from backend.environment.environment import CodeReviewEnvironment
    print("✅ CodeReviewEnvironment imported successfully")

except ImportError as e:
    print(f"❌ Failed to import CodeReviewEnvironment: {e}")
    sys.exit(1)

# ===== TEST 4: Runtime task validation =====
print("\n" + "=" * 60)
print("📋 TEST 4: Runtime task validation...")

try:
    env = CodeReviewEnvironment(use_dynamic_snippets=False)
    print("\n✅ Environment created successfully!")

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

        if grader is None:
            print("     ❌ ERROR: No grader attached!")

    print("\n✅ Runtime validation complete!")

except Exception as e:
    print(f"❌ Runtime error: {e}")

# ===== TEST 5: Correct file structure =====
print("\n" + "=" * 60)
print("📋 TEST 5: Checking correct file structure...")

required_files = [
    'openenv.yaml',
    'Dockerfile',
    'inference.py',
    'backend/__init__.py',
    'backend/environment/__init__.py',
    'backend/environment/environment.py',
    'backend/environment/models.py',
    'backend/environment/tasks.py',
    'backend/app.py',
]

missing_files = []

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - MISSING")
        missing_files.append(file_path)

# ===== FINAL SUMMARY =====
print("\n" + "=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)

errors = []

if len(tasks) < 3:
    errors.append("Less than 3 tasks defined")

if len(imported_graders) < 3:
    errors.append("Not all graders imported")

if len(missing_files) > 0:
    errors.append("Missing required files")

if errors:
    print("\n❌❌❌ ISSUES DETECTED ❌❌❌")
    for err in errors:
        print(f"- {err}")

    print("\n🔧 Fix these issues before submission!")
else:
    print("\n✅✅✅ ALL CHECKS PASSED (STRICT MODE) ✅✅✅")
    print("\nYour project is fully aligned with OpenEnv expectations.")
    print("You should be good to submit 🚀")

print("\n" + "=" * 60)
# aded dummy commit