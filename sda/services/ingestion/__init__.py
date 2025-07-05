# sda/services/ingestion/__init__.py

"""
This package handles the multi-stage process of ingesting code repositories,
including parsing, chunking, embedding, and persisting data.
"""

# Make the main service class available for import from this package
from .pipeline import IntelligentIngestionService

__all__ = ['IntelligentIngestionService']
