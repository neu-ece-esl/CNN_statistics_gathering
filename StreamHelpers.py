
from typing import (
    Any,
)
from copy import deepcopy
import ast
import re

class NamedEntityAnnotator(ast.NodeVisitor):
    def __init__(self, _ignore) -> None:
        self.ignore = _ignore
        super().__init__()

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.ignore:
            node.id = f"{{{node.id}}}"
        self.generic_visit(node)

    @classmethod
    def annotate(cls, node, ignore=[]):
        _node = deepcopy(node)
        annotator = cls(ignore)
        annotator.visit(_node)
        return _node


class NamedEntityExtractor(ast.NodeVisitor):
    def __init__(self, _ignore) -> None:
        self.entity_set = set()
        self.ignore = _ignore
        super().__init__()

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.ignore:
            self.entity_set.add(node.id)
        self.generic_visit(node)

    @classmethod
    def extract(cls, node, ignore=[]):
        extractor = cls(ignore)
        extractor.visit(node)
        return extractor.entity_set


class ISLGeneratorPreprocessor:
    @staticmethod
    def get_string_repr(s):
        return s.__str__()

    @staticmethod
    def remove_brackets(s):
        return s[1:-1]

    @staticmethod
    def remove_single_quotes(s):
        return re.sub("'", "", s)

    @staticmethod
    def preprocess_iterable(it):
        return ISLGeneratorPreprocessor.remove_single_quotes(
            ISLGeneratorPreprocessor.remove_brackets(ISLGeneratorPreprocessor.get_string_repr(it))
        )