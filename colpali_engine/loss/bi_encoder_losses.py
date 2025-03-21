import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss
from colpali_engine.utils.distributed import gather_with_grad


class BiEncoderLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            gathered_docs = doc_embeddings
        else:
            gathered_docs = gather_with_grad(doc_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, gathered_docs)
        labels = torch.arange(scores.shape[0], device=scores.device)
        rank = dist.get_rank() if dist.is_initialized() else 0
        num_logits = scores.shape[0]
        labels = labels + rank * num_logits

        loss_rowwise = self.ce_loss(scores / self.temperature, labels) * dist.get_world_size()

        return loss_rowwise


class BiPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        pos_scores = scores.diagonal()
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6
        neg_scores = neg_scores.max(dim=1)[0]

        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss


class BiPairwiseNegativeCELoss(torch.nn.Module):
    def __init__(self, in_batch_term=False):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        neg_doc_embeddings: (batch_size, dim)
        """

        # Compute the ColBERT scores
        pos_scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings).diagonal()
        neg_scores = torch.einsum("bd,cd->bc", query_embeddings, neg_doc_embeddings).diagonal()

        loss = F.softplus(neg_scores - pos_scores).mean()

        if self.in_batch_term:
            scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)

            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()

        return loss / 2
