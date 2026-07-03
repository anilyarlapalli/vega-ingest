"""Format parsers: bytes → ``DocumentModel``."""

from vega.parsers.base import Parser
from vega.parsers.image import ImageParser
from vega.parsers.pdf import PDFParser
from vega.parsers.text import TextParser

__all__ = ["Parser", "PDFParser", "ImageParser", "TextParser"]
