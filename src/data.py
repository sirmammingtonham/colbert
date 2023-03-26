from tqdm import tqdm
from datasets import DatasetDict, Dataset, load_dataset

MASK_PADDING = " ".join(["[MASK]"] * 32)


def load_eval_files(qrels_path, topK_path):
    queries = {}
    topK_docs = {}
    topK_pids = {}
    qrels = {}

    print("loading qrels")
    with open(qrels_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f):
            qid, _, pid, _ = [int(x) for x in line.strip().split("\t")]
            qrels[qid] = qrels.get(qid, [])
            qrels[qid].append(pid)

    print("loading top1000 ranks")
    with open(topK_path) as f:
        for line in tqdm(f):
            qid, pid, query, passage = line.split("\t")
            qid, pid = int(qid), int(pid)

            assert (qid not in queries) or (queries[qid] == query)
            queries[qid] = query
            topK_docs[qid] = topK_docs.get(qid, [])
            topK_docs[qid].append(passage)
            topK_pids[qid] = topK_pids.get(qid, [])
            topK_pids[qid].append(pid)

        print()

    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)
    print(len(queries), "unique queries")

    return queries, topK_docs, topK_pids, qrels


def get_train_dataset(tokenizer, train_path):
    def preprocess_function(examples):
        """
        * tokenizes dataset, each query in a batch gets a positive and negative
        * stacking positive and negative in the same batch for efficiency
        * reminder: technically halves effective batch size!
        """
        # "we pad [query] with BERT’s special [mask] tokens up to length Nq"
        # add maxlen mask tokens and let tokenizer handle truncation
        queries = [f"[CLS] [Q] {text} {MASK_PADDING}" for text in examples["query"]]
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

    ds = load_dataset(
        "csv",
        split="train",
        delimiter="\t",
        column_names=["query", "positive", "negative"],
        data_files=train_path,
    )
    ds = ds.map(
        preprocess_function,
        remove_columns=["query", "positive", "negative"],
        batched=True,
        batch_size=10000,
        num_proc=8,
    )
    ds.set_format(type="torch")
    return ds


def get_val_dataset(tokenizer, qrels_path, topK_path):
    def preprocess_function(examples):
        """
        * tokenizes dataset, each query in a batch gets a positive and negative
        * stacking positive and negative in the same batch for efficiency
        * reminder: technically halves effective batch size!
        """
        queries = [f"[CLS] [Q] {examples['query']} {MASK_PADDING}"]
        query_tokenized = tokenizer(
            queries,
            max_length=32,
            truncation=True,
            add_special_tokens=False,
        )

        docs = [f"[CLS] [D] {text}" for text in examples["top_docs"]]
        doc_tokenized = tokenizer(
            docs,
            max_length=128,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )

        # this will get broadcasted
        outputs = {f"query_{key}": value for key, value in query_tokenized.items()}
        outputs.update({f"doc_{key}": value for key, value in doc_tokenized.items()})

        return {"model_inputs": outputs}

    queries, topK_docs, topK_pids, qrels = load_eval_files(qrels_path, topK_path)
    ds = {"qid": [], "query": [], "top_docs": [], "top_doc_pids": [], "qrels": []}
    for qid in queries.keys():
        ds["qid"].append(qid)
        ds["query"].append(queries[qid])
        ds["top_docs"].append(topK_docs[qid])
        ds["top_doc_pids"].append(topK_pids[qid])
        ds["qrels"].append(qrels[qid])

    ds = Dataset.from_dict(ds)
    ds = ds.map(preprocess_function, batched=False, num_proc=8)
    ds.set_format(type="torch")
    return ds


def get_dataset(tokenizer, train_path, qrels_val_path, topK_val_path):
    return DatasetDict(
        {
            "train": get_train_dataset(tokenizer, train_path),
            "valid": get_val_dataset(tokenizer, qrels_val_path, topK_val_path),
        }
    )
