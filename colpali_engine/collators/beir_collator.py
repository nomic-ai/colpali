from random import randint
from typing import Any, Dict, List, Optional

from PIL import Image

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

import random


class BEIRCollator(VisualRetrieverCollator):
    """
    Collator for BEIR-style dataset training.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        corpus: Optional["Dataset"] = None,  # noqa: F821
    ):
        super().__init__(processor=processor, max_length=max_length)
        if corpus is None:
            raise ValueError("Corpus is required for BEIRCollator")
        self.corpus = corpus

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_examples: List[Dict[str, Any]] = []
        for example in examples:
            # Extract positive document id depending on the corpus format.
            positive_docs = example["positive_docs"]
            pos_docid = random.choice([doc_id for (doc_id, score) in positive_docs if score > 0])

            sample = {
                "image": self.corpus[pos_docid]["image"],
                "query": example["query"],
            }

            processed_examples.append(sample)

        return super().__call__(processed_examples)
