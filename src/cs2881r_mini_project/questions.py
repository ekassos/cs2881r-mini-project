from datasets import load_dataset
import json


def get_long_wikiqa_questions(n=230, split="train"):
    """
    Fetches longest n questions from the WikiQA split dataset.
    """
    dataset = load_dataset("microsoft/wiki_qa", split=split)

    # Use a set to track seen question_ids
    seen_ids = set()
    unique_questions = []

    # Iterate through dataset and collect unique questions
    for example in dataset:
        qid = example["question_id"]
        if qid not in seen_ids:
            seen_ids.add(qid)
            unique_questions.append(example["question"])
        if len(unique_questions) == n:
            break

    # Save to JSONL file with an index field "Q"
    output_path = "data/questions.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, question in enumerate(unique_questions, start=1):
            json.dump({"id": f"{idx}", "question": question}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    questions = get_long_wikiqa_questions(n=230, split="train")
