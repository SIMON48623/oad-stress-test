from pathlib import Path
import sys

SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUPPLEMENT_ROOT))
