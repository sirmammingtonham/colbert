from datasets import load_dataset

def get_dataset(data_files, tokenizer):
    dataset = load_dataset("tsv", data_files=data_files)

    mask_padding = " ".join(["[MASK]"] * 32)

    def preprocess_function(examples):
        """
        * tokenizes dataset, each query in a batch gets a positive and negative
        * stacking positive and negative in the same batch for efficiency
        * reminder: technically halves effective batch size!
        """
        # "we pad [query] with BERT’s special [mask] tokens up to length Nq"
        # add maxlen mask tokens and let tokenizer handle truncation
        queries = [f"[CLS] [Q] {text} {mask_padding}" for text in examples["query"]]
        query_tokenized = tokenizer(
            queries,
            max_length=32,
            truncation=True,
            add_special_tokens=False,
        )

        # "Unlike queries, we do not append [mask] tokens to documents."
        doc_positives = [f"[CLS] [D] {text}" for text in examples["positive"]]
        doc_negatives = [f"[CLS] [D] {text}" for text in examples["negative"]]
        doc_tokenized = tokenizer(
            doc_positives + doc_negatives,
            max_length=128,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )

        # stack queries along batch so each query gets a positive and negative
        outputs = {f"query_{key}": value * 2 for key, value in query_tokenized.items()}
        outputs.update({f"doc_{key}": value for key, value in doc_tokenized.items()})

        return outputs

    dataset = dataset.map(preprocess_function, batched=True)
    dataset.set_format(type="torch")
    return dataset
