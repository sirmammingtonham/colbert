from dataclasses import dataclass

from src.data import get_dataset
from src.model import ColBERT
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments

SEED = 808

data_files = {
    "train": "train.csv",
    "valid": "qrels.dev.small.tsv",
    "test": "test.csv",
}


def main(data_args, training_args, model_str="bert_base_uncased"):
    bert = BertModel.from_pretrained(model_str)
    tokenizer = BertTokenizerFast.from_pretrained(model_str)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    datasets = get_dataset(data_args.data_files, tokenizer)
    model = ColBERT(bert, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=None,
    )

    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    eval_results = trainer.evaluate(datasets["test"])

    return train_results, eval_results


if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="./models",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-6,
        logging_dir="./logs",
        seed=SEED,
    )
    main(training_args)
