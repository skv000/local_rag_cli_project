import csv
import json
from pathlib import Path
from typing import List, Union

from .document import Document


class DocumentLoader:
    """Loads text and structured documents from files or directories."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".py", ".html"}

    @classmethod
    def load_file(cls, file_path: Union[str, Path]) -> Document:
        """Load a single document from a file path."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix in {".txt", ".md", ".markdown", ".py", ".html"}:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, str):
                    content = data
                else:
                    content = json.dumps(data, indent=2, ensure_ascii=False)
        elif suffix == ".csv":
            lines = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append(", ".join(row))
            content = "\n".join(lines)
        else:
            # Fallback text reading
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        return Document(
            content=content,
            source=str(path),
            metadata={
                "filename": path.name,
                "file_extension": suffix,
                "file_size": path.stat().st_size,
            },
        )

    @classmethod
    def load_directory(
        cls,
        directory_path: Union[str, Path],
        recursive: bool = True,
    ) -> List[Document]:
        """Loads all supported files from a directory."""
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        documents: List[Document] = []
        iterator = dir_path.rglob("*") if recursive else dir_path.glob("*")

        for file_path in iterator:
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    doc = cls.load_file(file_path)
                    if doc.content.strip():
                        documents.append(doc)
                except Exception as e:
                    print(f"[Warning] Failed to load {file_path}: {e}")

        return documents
