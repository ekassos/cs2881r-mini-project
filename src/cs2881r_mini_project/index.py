import json
from pathlib import Path
import bm25s
import Stemmer


def load_jsonl_basic(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def chunk_text(text, max_length=256, stride=128):
    """
    Splits text into overlapping chunks of `max_length` tokens
    with an overlap of `max_length - stride`.
    """
    tokens = text.split()  # simple whitespace tokenization
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_length
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start += stride  # slide forward
    return chunks


def main(
    data_dir="data",
    index_dir="bm25s_indices/",
    dataset="articles",
    max_length=256,
    stride=128,
):
    index_dir = Path(index_dir) / dataset
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the article corpus...")
    corpus = load_jsonl_basic(Path(data_dir) / f"{dataset}.jsonl")

    corpus_records = []
    for k, v in enumerate(corpus):
        text = v.get("extract", "")
        chunks = chunk_text(text, max_length=max_length, stride=stride)
        for i, chunk in enumerate(chunks):
            corpus_records.append({"id": f"{k}_{i}", "text": chunk})

    corpus_lst = [r["text"] for r in corpus_records]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus_lst, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25(corpus=corpus_records, backend="numba", method="robertson")
    retriever.index(corpus_tokens)

    retriever.save(str(index_dir))
    print(f"Saved the index to {index_dir}.")


if __name__ == "__main__":
    main()
