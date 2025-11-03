import os
import json
import argparse


def count_total_words(path):
    """
    Reads a JSONL file of Wikipedia articles and computes:
    - Total word count across all articles
    - Number of articles
    - Average words per article
    """
    total_words = 0
    num_articles = 0

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
                wc = article.get("words")
                if wc is None and "extract" in article:
                    wc = len(article["extract"].split())
                if wc:
                    total_words += wc
                    num_articles += 1
            except json.JSONDecodeError:
                print("⚠️ Skipping malformed line.")
                continue

    if num_articles == 0:
        print("⚠️ No valid articles found.")
        return

    avg_words = total_words / num_articles
    print(f"📚 Processed {num_articles} articles")
    print(f"🧮 Total words: {total_words:,}")
    print(f"📏 Average words per article: {avg_words:.1f}")

    return {"articles": num_articles, "total_words": total_words, "average": avg_words}


def load_prompts(data_path="data/prompts.jsonl", test_mode=False):
    prompts = []
    with open(data_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    if test_mode:
        prompts = prompts[:5]
    return prompts


def append_to_jsonl(file_path, data):
    with open(file_path, "a") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_to_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_references(
    ref_path="data/references.jsonl", prompt_path="data/prompts.jsonl", test_mode=False
):
    references = {}
    with open(ref_path, "r") as f:
        for line in f:
            item = json.loads(line)
            references[item["id"]] = item["reference"]
    return references


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count total words from a JSONL file of Wikipedia articles."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="data/articles.jsonl",
        help="Path to the JSONL file (default: data/articles.jsonl)",
    )
    args = parser.parse_args()

    count_total_words(args.file)
