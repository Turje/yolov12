"""
Weights & Biases Setup
======================

Helper script to configure W&B for YOLOv12 training.

Usage:
    import scripts.wandb_setup as wandb_setup
    wandb_setup.init_wandb()
"""

import os
import wandb


def init_wandb(project_name="vizwiz_yolov12", api_key=None):
    """
    Initialize Weights & Biases.
    
    Args:
        project_name: W&B project name (default: vizwiz_yolov12)
        api_key: W&B API key (if None, reads from WANDB_API_KEY env var)
    
    Returns:
        None
    """
    if api_key is None:
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key is None:
            raise ValueError(
                "WANDB_API_KEY not found in environment variables. "
                "Please set it: export WANDB_API_KEY=your_key"
            )
    
    wandb.login(key=api_key)
    print(f"W&B initialized for project: {project_name}")
    return project_name


def log_model_artifact(run, model_path, artifact_name="best_model", artifact_type="model"):
    """
    Log a model checkpoint as a W&B artifact.
    
    Args:
        run: W&B run object
        model_path: Path to model checkpoint file
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (default: "model")
    
    Returns:
        W&B artifact
    """
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print(f"Logged artifact: {artifact_name} from {model_path}")
    return artifact


def log_metrics(run, metrics_dict, step=None):
    """
    Log metrics to W&B.
    
    Args:
        run: W&B run object
        metrics_dict: Dictionary of metric names to values
        step: Optional step number
    """
    if step is not None:
        run.log(metrics_dict, step=step)
    else:
        run.log(metrics_dict)
    
    # Also update summary for final metrics
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            run.summary[key] = value

