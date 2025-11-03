import bm25s
import Stemmer
import os


def ask(question, index_dir="bm25s_indices/", dataset="articles"):
    """
    Given a question, retrieve the most relevant chunk from the BM25 index.
    """
    save_dir = f"{index_dir}/{dataset}"

    if not os.path.exists(save_dir):
        raise FileNotFoundError(
            f"Index directory {save_dir} does not exist. Please build the index first by running `python index.py`."
        )

    reloaded_retriever = bm25s.BM25.load(save_dir, load_corpus=True)
    stemmer = Stemmer.Stemmer("english")

    query_tokens = bm25s.tokenize(question, stemmer=stemmer)

    results, _ = reloaded_retriever.retrieve(query_tokens, k=1)
    if results:
        return results[0][0]["id"], results[0][0]["text"]
    else:
        return "No relevant information found."


if __name__ == "__main__":
    answer = ask(
        "How are the directions of the velocity and force vectors related in a circular motion?"
    )
    print("Answer:", answer)
