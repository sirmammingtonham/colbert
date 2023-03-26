import string
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from typing import Optional


class ColBERT(nn.Module):
    def __init__(
        self,
        bert_model: BertModel,
        tokenizer: BertTokenizer,
        feature_dim: int = 128,
    ):
        super().__init__()

        self.bert = bert_model
        self.bert.resize_token_embeddings(len(tokenizer))  # since we added [Q], [D]

        self.linear = nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=feature_dim,
            bias=False,
        )

        self.skip_list = {
            tokenizer.encode(symbol, add_special_tokens=False)[0]
            for symbol in string.punctuation
        }

    def pairwise_maxsim(self, query_outputs, doc_outputs):
        sims = query_outputs @ doc_outputs.mT # torch.einsum("bqh,bdh->bqd", query_outputs, doc_outputs)
        scores = sims.max(-1).values.sum(-1)
        return scores

    def loss(self, scores):
        # split superbatch into corresponding positive/negative pairs (B*2) -> (B, 2)
        logits = scores.view(2, -1).permute(1, 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        query_token_type_ids: Optional[torch.Tensor] = None,
        doc_input_ids: Optional[torch.Tensor] = None,
        doc_attention_mask: Optional[torch.Tensor] = None,
        doc_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:

        query_output = self.bert(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            **kwargs,
        )
        query_embedding = self.linear(query_output.last_hidden_state)
        query_embedding = F.normalize(input=query_embedding, p=2, dim=2)

        doc_output = self.bert(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=doc_token_type_ids,
            **kwargs,
        )
        doc_embedding = self.linear(doc_output.last_hidden_state)

        # filter (punctuation masking)
        mask = [
            [(token not in self.skip_list) and (token != 0) for token in doc]
            for doc in doc_input_ids.cpu().tolist()
        ]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(-1)
        doc_embedding = doc_embedding * mask
        doc_embedding = F.normalize(input=doc_embedding, p=2, dim=2)
        #

        scores = self.pairwise_maxsim(query_embedding, doc_embedding)
        loss = self.loss(scores)
        return {
            "score": scores,
            "loss": loss,
            "query_embed": query_embedding,
            "doc_embed": doc_embedding,
        }
