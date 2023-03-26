import os
from src.metrics import EvalMetricCallback
from src.data import get_dataset
from src.model import ColBERT
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_from_disk

SEED = 808


def main(dataset_path, data_files, training_args, model_str="bert-base-uncased"):
    bert = BertModel.from_pretrained(model_str)
    tokenizer = BertTokenizerFast.from_pretrained(model_str)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    if os.path.isfile(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = get_dataset(tokenizer, **data_files)
        ds.save_to_disk(dataset_path)

    model = ColBERT(bert, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        callbacks=[EvalMetricCallback(model, ds["valid"])],
        tokenizer=tokenizer,
        data_collator=None,
        report_to="wandb",
    )

    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    print(train_results)
    # eval_results = trainer.evaluate(ds["test"])
    # print(eval_results)

    return train_results


if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="./models",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-6,
        logging_dir="./logs",
        push_to_hub=True,
        seed=SEED,
    )
    dataset_path = "data/dataset.hf"
    data_files = {
        "train_path": "data/triples.train.small.tsv",
        "qrels_val_path": "data/qrels.dev.small.tsv",
        "topK_val_path": "data/top1000.dev",
    }
    main(dataset_path, data_files, training_args)
