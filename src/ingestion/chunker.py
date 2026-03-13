"""
Chunking utilities for GDScript files and Godot documentation.

Provides functions to split GDScript code, HTML documentation, and Q&A data
into smaller chunks suitable for embedding and retrieval.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
from bs4 import BeautifulSoup

from src import config


@dataclass
class Chunk:
    """Represents a text chunk with metadata for embedding and retrieval."""

    text: str
    source: str
    source_type: str
    metadata: dict


def chunk_gdscript_file(file_path: Path, source_name: str) -> Iterator[Chunk]:
    """
    Chunk a GDScript file into smaller pieces with overlapping context.

    Splits code at function or class boundaries, maintaining up to
    CHUNK_MAX_LINES per chunk with CHUNK_OVERLAP_LINES lines of overlap.

    Args:
        file_path: Path to the GDScript file.
        source_name: Name of the source repository or file.

    Yields:
        Chunk objects containing the text, source, source_type, and metadata.
    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.split("\n")

    current_chunk_lines = []
    current_chunk_start = 0

    function_name = None

    for i, line in enumerate(lines):
        func_match = re.match(r"^func\s+(\w+)", line.strip())
        class_match = re.match(r"^class\s+(\w+)", line.strip())

        if func_match:
            function_name = func_match.group(1)
        elif class_match:
            function_name = class_match.group(1)

        current_chunk_lines.append(line)

        if len(current_chunk_lines) >= config.CHUNK_MAX_LINES:
            text = "\n".join(current_chunk_lines).strip()
            if text:
                yield Chunk(
                    text=text,
                    source=source_name,
                    source_type="gdscript_repo",
                    metadata={
                        "file": str(file_path),
                        "start_line": current_chunk_start + 1,
                        "function": function_name,
                    },
                )

            overlap_lines = current_chunk_lines[-config.CHUNK_OVERLAP_LINES:]
            current_chunk_lines = overlap_lines.copy()
            current_chunk_start = i - len(overlap_lines) + 1

    if current_chunk_lines:
        text = "\n".join(current_chunk_lines).strip()
        if text:
            yield Chunk(
                text=text,
                source=source_name,
                source_type="gdscript_repo",
                metadata={
                    "file": str(file_path),
                    "start_line": current_chunk_start + 1,
                    "function": function_name,
                },
            )


def chunk_gdscript_repo(repo_file: Path) -> Iterator[Chunk]:
    """
    Chunk a GDScript repository file containing multiple code snippets.

    Parses a repository file from the HuggingFace dataset, extracting individual
    GDScript files and chunking them. Skips non-Godot 4 repositories.

    Args:
        repo_file: Path to the repository file from the dataset.

    Yields:
        Chunk objects for each code snippet in the repository.
    """
    content = repo_file.read_text(encoding="utf-8", errors="ignore")

    repo_name = repo_file.stem

    version_pattern = r"### Godot version:\s*(\d+)|Godot version:\s*(\d+)"
    version_match = re.search(version_pattern, content)
    version = version_match.group(1) or version_match.group(2) if version_match else None
    if version != "4":
        print(f"Skipping {repo_name}: not Godot 4")
        return

    pattern = r"### Files:\s*File name: (.*?)\n```(?:gdscript|markdown)?\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)

    for file_name, code in matches:
        code = code.strip()
        if not code:
            continue

        temp_file_path = repo_file.parent / f"temp_{file_name}.gd"
        temp_file_path.write_text(code)

        yield from chunk_gdscript_file(temp_file_path, repo_name)

        temp_file_path.unlink(missing_ok=True)


def chunk_html_file(file_path: Path, source_name: str) -> Iterator[Chunk]:
    """
    Chunk a single HTML file from Godot documentation.

    Extracts sections marked by headings (h1, h2, h3) and creates chunks
    from each section's content.

    Args:
        file_path: Path to the HTML file.
        source_name: Name identifying the documentation source.

    Yields:
        Chunk objects for each section in the HTML file.
    """
    html_content = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_content, "lxml")

    for section in soup.find_all(["section", "div"]):
        h_tag = section.find(["h1", "h2", "h3"])
        if not h_tag:
            continue

        heading = h_tag.get_text(strip=True)
        section_text = section.get_text(separator="\n", strip=True)

        if len(section_text) < 50:
            continue

        text = f"## {heading}\n\n{section_text}"

        yield Chunk(
            text=text,
            source=source_name,
            source_type="godot_docs",
            metadata={
                "file": str(file_path),
                "heading": heading,
            },
        )


def chunk_godot_docs(docs_dir: Path) -> Iterator[Chunk]:
    """
    Chunk all HTML files in the Godot documentation directory.

    Recursively finds all HTML files and processes them using chunk_html_file.

    Args:
        docs_dir: Path to the extracted Godot documentation directory.

    Yields:
        Chunk objects for each section in all HTML files.
    """
    html_files = list(docs_dir.rglob("*.html"))
    print(f"Found {len(html_files)} HTML files in {docs_dir}")

    for html_file in html_files:
        try:
            relative_path = html_file.relative_to(docs_dir)
            source_name = str(relative_path.parent / relative_path.stem)
            yield from chunk_html_file(html_file, source_name)
        except Exception as e:
            print(f"Error processing {html_file}: {e}")


def chunk_godot_qa(data_dir: Path) -> Iterator[Chunk]:
    """
    Chunk Godot Q&A data from the Glaive AI dataset.

    Reads the JSON file containing prompt-response pairs and creates
    chunks from each Q&A entry.

    Args:
        data_dir: Path to the directory containing the Q&A JSON file.

    Yields:
        Chunk objects for each Q&A pair.
    """
    import json

    json_file = data_dir / "data" / "glaive_godot4_docs.json"
    if not json_file.exists():
        print(f"QA file not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    for item in qa_data:
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        if not prompt or not response:
            continue

        text = f"Q: {prompt}\n\nA: {response}"

        yield Chunk(
            text=text,
            source="godot_4_docs",
            source_type="qa",
            metadata={},
        )
