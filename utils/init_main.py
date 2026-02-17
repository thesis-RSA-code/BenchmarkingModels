import os


def is_running_in_container():
    """Check if running inside Singularity/Apptainer container."""
    if os.getenv('SINGULARITY_NAME') or os.getenv('SINGULARITY_CONTAINER') or os.getenv('APPTAINER_CONTAINER'):
        return True
    return False