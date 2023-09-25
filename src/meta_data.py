"""project related dir as env var
"""
import os
from typing import Dict

def get_meta_data() -> Dict[str, str]:
    """specify project dirs

    Returns:
        Dict[str, str]: directory lookup
    """
    meta_data = {
        'SEC_DIR': os.environ['SEC_DIR'], # intermeidate step and output
        'LOG_DIR': os.environ['LOG_DIR'],
        'MODEL_DIR': os.environ['MODEL_DIR']
    }

    return meta_data