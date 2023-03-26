import torch
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)


def rerank(colbert, top_doc_pids, **model_inputs):
    scores = colbert(**model_inputs)["scores"].cpu()

    ranked = scores.argsort(descending=True)
    reranked_pids = top_doc_pids[ranked]
    return reranked_pids.tolist()


class ColBERTMetrics:
    def __init__(self, recall_depths=[50, 200, 1000]):
        self.num_queries = 0
        self.mrr10_sum = 0.0
        self.recall_sums = {depth: 0.0 for depth in recall_depths}

    def add(self, ranking: torch.Tensor, gold_positives: torch.Tensor):
        self.num_queries += 1
        positives = [
            i for i, pid in enumerate(ranking) if pid in set(gold_positives.tolist())
        ]

        if len(positives) == 0:
            return

        self.mrr10_sum += (1.0 / (positives[0] + 1.0)) if positives[0] < 10 else 0.0

        for depth in self.recall_sums.keys():
            num_positives = len([pos for pos in positives if pos < depth])
            self.recall_sums[depth] += num_positives / len(gold_positives)

    def add_batch(self, batch_ranking: torch.Tensor, batch_positives: torch.Tensor):
        for ranking, gold_positives in zip(batch_ranking, batch_positives):
            self.add(ranking, gold_positives)

    def compute(self):
        output = {}
        score = self.mrr10_sum / self.num_queries
        output["mrr@10"] = score
        print(f"MRR@10 == {score}")

        for depth in sorted(self.recall_sums):
            score = self.recall_sums[depth] / self.num_queries
            output[f"recall@{depth}"] = score
            print(f"Recall@{depth} == {score}")

        return output


class EvalMetricCallback(TrainerCallback):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        metric = ColBERTMetrics()

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        for example in iter(dataloader):
            # get rid of extra batch dimension since we prebatched during tokenization
            model_inputs = example.pop("model_inputs")
            model_inputs = {
                key: value[0].to(args.device) for key, value in model_inputs.items()
            }
            example = {key: value[0] for key, value in example.items()}
            ranking = rerank(self.model, example["top_doc_pids"], **model_inputs)
            metric.add(ranking, example["qrels"])

        result = metric.compute()

        if self._wandb is not None:
            self._wandb.log({**result, "step": state.global_step})
