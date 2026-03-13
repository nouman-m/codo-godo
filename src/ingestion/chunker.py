"""
Chunking utilities for GDScript files and Godot documentation.

Provides functions to split GDScript code, RST documentation, and Q&A data
into smaller chunks suitable for embedding and retrieval.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

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


def chunk_rst_file(file_path: Path, source_name: str) -> Iterator[Chunk]:
    """
    Chunk a single RST file from Godot documentation.

    Extracts sections, descriptions, and code examples from RST files,
    stripping RST directives and formatting.

    Args:
        file_path: Path to the RST file.
        source_name: Name identifying the documentation source.

    Yields:
        Chunk objects for each section in the RST file.
    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
        return

    content = re.sub(r"^\.\. \w+::.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\.\. \w+$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^:github_url:.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^:.*?:.*?$", "", content, flags=re.MULTILINE)

    lines = content.split("\n")
    current_section = []
    current_heading = None
    code_block_content = []

    for i, line in enumerate(lines):
        underline_match = re.match(r"^=+$|^-+$|~+$|\^+$", line.strip())
        if underline_match and i > 0 and len(lines[i-1].strip()) > 0 and len(line.strip()) >= len(lines[i-1].strip()):
            if current_section or code_block_content:
                section_text = "\n".join(current_section).strip()
                code_text = "\n".join(code_block_content).strip()

                parts = []
                if current_heading:
                    parts.append(f"## {current_heading}")
                if section_text:
                    parts.append(section_text)
                if code_text:
                    parts.append(f"\n```gdscript\n{code_text}\n```")

                if parts:
                    text = "\n".join(parts)
                    if len(text) > 100:
                        yield Chunk(
                            text=text,
                            source=source_name,
                            source_type="godot_docs",
                            metadata={
                                "file": str(file_path),
                                "heading": current_heading or "",
                            },
                        )

            current_heading = lines[i-1].strip()
            current_section = []
            code_block_content = []

        elif line.strip().startswith(".. code-tab::"):
            continue
        elif line.startswith("    ") or line.startswith("\t"):
            code_block_content.append(line.strip())
        else:
            if code_block_content:
                code_text = "\n".join(code_block_content).strip()
                if code_text and current_heading:
                    yield Chunk(
                        text=f"## {current_heading}\n\n```gdscript\n{code_text}\n```",
                        source=source_name,
                        source_type="godot_docs",
                        metadata={
                            "file": str(file_path),
                            "heading": current_heading,
                        },
                    )
                code_block_content = []
            if line.strip() and not line.startswith(".. "):
                current_section.append(line)

    if current_section or code_block_content:
        section_text = "\n".join(current_section).strip()
        code_text = "\n".join(code_block_content).strip()

        parts = []
        if current_heading:
            parts.append(f"## {current_heading}")
        if section_text:
            parts.append(section_text)
        if code_text:
            parts.append(f"\n```gdscript\n{code_text}\n```")

        if parts:
            text = "\n".join(parts)
            if len(text) > 100:
                yield Chunk(
                    text=text,
                    source=source_name,
                    source_type="godot_docs",
                    metadata={
                        "file": str(file_path),
                        "heading": current_heading or "",
                    },
                )


def chunk_godot_docs(docs_dir: Path) -> Iterator[Chunk]:
    """
    Chunk all RST files in the Godot documentation _sources directory.

    Finds all .rst.txt files in the _sources subdirectory and processes them
    using chunk_rst_file.

    Args:
        docs_dir: Path to the extracted Godot documentation directory.

    Yields:
        Chunk objects for each section in all RST files.
    """
    sources_dir = docs_dir / "_sources"
    if not sources_dir.exists():
        print(f"_sources directory not found in {docs_dir}")
        return

    rst_files = list(sources_dir.rglob("*.rst.txt"))
    print(f"Found {len(rst_files)} RST files in {sources_dir}")

    for rst_file in rst_files:
        try:
            if rst_file.name in ("404.rst.txt",):
                continue

            relative_path = rst_file.relative_to(sources_dir)
            source_name = str(relative_path.parent / relative_path.stem)
            yield from chunk_rst_file(rst_file, source_name)
        except Exception as e:
            print(f"Error processing {rst_file}: {e}")


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
