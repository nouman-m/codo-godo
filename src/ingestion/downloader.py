"""
Data downloading utilities for HuggingFace datasets.

Provides functions to download GDScript repositories and Godot Q&A data
from HuggingFace Hub.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

from src import config


def download_gdscript_repos(output_dir: Path | None = None) -> Path:
    """
    Download GDScript repository files from HuggingFace.

    Downloads all .txt files from the wallstoneai/godot-gdscript-dataset dataset.
    Skips files that already exist locally.

    Args:
        output_dir: Optional custom output directory. Defaults to RAW_DATA_DIR/gdscript_repos.

    Returns:
        Path: The directory where files were downloaded.
    """
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR / "gdscript_repos"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GDScript repos to {output_dir}...")

    repo_id = config.HUGGINGFACE_DATASETS["gdscript_repos"]
    files = list(list_repo_files(repo_id, repo_type="dataset"))

    txt_files = [f for f in files if f.endswith(".txt")]

    print(f"Found {len(txt_files)} .txt files to download")

    for i, file_path in enumerate(txt_files):
        dest_path = output_dir / "files" / Path(file_path).name
        if not dest_path.exists():
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=output_dir,
            )
        if (i + 1) % 500 == 0:
            print(f"Downloaded {i + 1}/{len(txt_files)} files...")

    print(f"Downloaded GDScript repos to {output_dir}")
    return output_dir


def download_godot_qa(output_dir: Path | None = None) -> Path:
    """
    Download Godot Q&A dataset from HuggingFace.

    Downloads the parquet file from glaiveai/godot_4_docs dataset.

    Args:
        output_dir: Optional custom output directory. Defaults to RAW_DATA_DIR/godot_qa.

    Returns:
        Path: The directory where the file was downloaded.
    """
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR / "godot_qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Godot Q&A to {output_dir}...")

    repo_id = config.HUGGINGFACE_DATASETS["godot_qa"]

    hf_hub_download(
        repo_id=repo_id,
        filename="data/train.parquet",
        repo_type="dataset",
        local_dir=output_dir,
    )

    print(f"Downloaded Godot Q&A to {output_dir}")
    return output_dir


def get_godot_docs_dir() -> Path:
    """
    Get the path to the Godot documentation directory.

    Returns:
        Path: Path to the godot-docs directory.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
    """
    if not config.GODOT_DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Godot docs not found at {config.GODOT_DOCS_DIR}. "
            f"Please download and extract the HTML docs there."
        )
    return config.GODOT_DOCS_DIR


def get_gdscript_repos_dir() -> Path:
    """
    Get the path to the downloaded GDScript repositories directory.

    Returns:
        Path: Path to the gdscript_repos directory.
    """
    return config.RAW_DATA_DIR / "gdscript_repos"


def get_godot_qa_dir() -> Path:
    """
    Get the path to the downloaded Godot Q&A directory.

    Returns:
        Path: Path to the godot_qa directory.
    """
    return config.RAW_DATA_DIR / "godot_qa"
