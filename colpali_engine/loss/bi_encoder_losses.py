import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class BiEncoderLoss(torch.nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.bidirectional = bidirectional

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)
        loss = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))

        if self.bidirectional:
            # transpose the scores matrix
            scores_t = scores.t()
            loss_colwise = self.ce_loss(scores_t, torch.arange(scores_t.shape[0], device=scores_t.device))
            loss = (loss + loss_colwise) / 2

        return loss

class BiEncoderInfoNCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss = NLLLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """
        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)
        # softmax
        scores = F.softmax(scores, dim=1)
        # keep only diag
        scores = torch.diagonal(scores)
        # loss is -log(diag)
        loss = -torch.log(scores).mean()
        return loss


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
