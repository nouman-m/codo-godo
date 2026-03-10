import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
from bs4 import BeautifulSoup

from src import config


@dataclass
class Chunk:
    text: str
    source: str
    source_type: str
    metadata: dict


def chunk_gdscript_file(file_path: Path, source_name: str) -> Iterator[Chunk]:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.split("\n")

    current_chunk_lines = []
    current_chunk_start = 0

    in_function = False
    function_name = None

    for i, line in enumerate(lines):
        func_match = re.match(r"^func\s+(\w+)", line.strip())
        class_match = re.match(r"^class\s+(\w+)", line.strip())

        if func_match:
            in_function = True
            function_name = func_match.group(1)
        elif class_match:
            in_function = True
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
    content = repo_file.read_text(encoding="utf-8", errors="ignore")

    repo_name = repo_file.stem

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
    import pandas as pd

    parquet_file = data_dir / "train.parquet"
    if not parquet_file.exists():
        print(f"QA file not found: {parquet_file}")
        return

    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        prompt = row.get("prompt", "")
        response = row.get("response", "")

        if not prompt or not response:
            continue

        text = f"Q: {prompt}\n\nA: {response}"

        yield Chunk(
            text=text,
            source="godot_4_docs",
            source_type="qa",
            metadata={},
        )
