import sys
import os

# Ensure the root directory is on the path so backend gets resolved correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our existing compliant grader classes using absolute mapping
from backend.environment.tasks import (
    BugDetectionGrader, 
    BugClassificationGrader, 
    FixSuggestionGrader
)

class EasyGrader(BugDetectionGrader):
    """Alias for BugDetectionGrader for OpenEnv compatibility"""
    pass

class MediumGrader(BugClassificationGrader):
    """Alias for BugClassificationGrader for OpenEnv compatibility"""
    pass

class HardGrader(FixSuggestionGrader):
    """Alias for FixSuggestionGrader for OpenEnv compatibility"""
    pass
