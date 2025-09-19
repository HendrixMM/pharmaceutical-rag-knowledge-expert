"""Canonical paper schema used by synthesis and DDI processors."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Union

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    """Normalized representation of a literature paper used by analysis engines."""

    page_content: str = Field(default="", alias="page_content")
    content: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    abstract: str | None = None
    pmid: str | None = None
    pk_parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - exercised indirectly
        if not isinstance(self.metadata, dict):
            self.metadata = {}

        # Normalise primary metadata fields
        for key in ("title", "abstract", "pmid"):
            value = getattr(self, key)
            if value is None:
                meta_value = self.metadata.get(key)
                if isinstance(meta_value, str):
                    setattr(self, key, meta_value)
            else:
                self.metadata.setdefault(key, value)

        if not self.page_content and self.content:
            self.page_content = self.content
        elif not self.page_content:
            self.page_content = ""

        if not self.pk_parameters:
            pk_meta = self.metadata.get("pk_parameters")
            if isinstance(pk_meta, dict):
                self.pk_parameters = dict(pk_meta)
            else:
                self.pk_parameters = {}
        else:
            self.pk_parameters = dict(self.pk_parameters)

        self.metadata.setdefault("pk_parameters", dict(self.pk_parameters))

    @property
    def text(self) -> str:
        """Return the best available textual content for downstream analysis."""
        return self.page_content or self.content or ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation with guaranteed schema fields."""
        metadata = dict(self.metadata)
        metadata.setdefault("title", self.title)
        metadata.setdefault("abstract", self.abstract)
        metadata.setdefault("pmid", self.pmid)
        metadata["pk_parameters"] = dict(self.pk_parameters)

        result: Dict[str, Any] = {
            "__paper_schema_validated__": True,
            "page_content": self.text,
            "content": self.content,
            "metadata": metadata,
        }

        extra = getattr(self, "model_extra", None)
        if extra:
            result.update(extra)

        if self.title is not None:
            result.setdefault("title", self.title)
        if self.abstract is not None:
            result.setdefault("abstract", self.abstract)
        if self.pmid is not None:
            result.setdefault("pmid", self.pmid)

        return result


def coerce_paper(entry: Union[Paper, Dict[str, Any]]) -> Paper:
    """Return a `Paper` instance for the supplied entry."""
    if isinstance(entry, Paper):
        return entry
    if isinstance(entry, dict):
        return Paper.model_validate(entry)
    raise TypeError("Paper entries must be mappings or Paper models.")


def coerce_papers(entries: Iterable[Union[Paper, Dict[str, Any]]]) -> list[Paper]:
    """Coerce a collection of paper-like entries into `Paper` instances."""
    return [coerce_paper(entry) for entry in entries]


__all__ = ["Paper", "coerce_paper", "coerce_papers"]

