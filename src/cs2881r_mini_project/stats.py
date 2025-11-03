import evaluate
import numpy as np
from nltk import word_tokenize, download
from dataclasses import dataclass, asdict
import json
from typing import List
import sacrebleu

download("punkt_tab", quiet=True)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")


@dataclass
class ModelMetrics:
    model_name: str
    reasoning_effort: str | None = None
    rouge_l: float | None = None
    rouge_l_std: float | None = None
    rouge_l_excluding_zero: float | None = None
    rouge_l_excluding_zero_std: float | None = None
    bleu: float | None = None
    bleu_std: float | None = None
    bleu_excluding_zero: float | None = None
    bleu_excluding_zero_std: float | None = None
    bertscore: float | None = None
    bertscore_std: float | None = None
    bertscore_excluding_zero: float | None = None
    bertscore_excluding_zero_std: float | None = None
    f1_score: float | None = None
    f1_score_std: float | None = None
    f1_score_excluding_zero: float | None = None
    f1_score_excluding_zero_std: float | None = None


def stats(arr):
    return np.mean(arr) * 100, np.std(arr) * 100


def compute_metrics(
    model_name: str,
    reasoning_effort: str | None,
    predictions: List[str],
    references: List[str],
    min_tokens: int | None = None,
):
    preds = [(p or "") for p in predictions]
    refs = [(r or "") for r in references]

    n = len(preds)
    non_empty = [p.strip() != "" for p in preds]
    pred_tokens_list = [
        word_tokenize(p.lower()) if ne else [] for p, ne in zip(preds, non_empty)
    ]
    ref_tokens_list = [word_tokenize(r.lower()) for r in refs]

    length_ok = [
        (min_tokens is None) or (len(toks) >= min_tokens) for toks in pred_tokens_list
    ]

    # ROUGE-L (batched, per-sample)
    rouge_res = rouge.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rougeL"],
        use_aggregator=False,
    )
    rouge_scores = rouge_res["rougeL"]

    # BLEU: compute sentence-level BLEU with sacrebleu for consistency and speed
    bleu_scores = [
        float(sacrebleu.sentence_bleu(p, [r]).score) / 100.0 if ne else 0.0
        for p, r, ne in zip(preds, refs, non_empty)
    ]

    # Token-level F1
    f1_scores = []
    for pred_toks, ref_toks in zip(pred_tokens_list, ref_tokens_list):
        if not pred_toks:
            f1_scores.append(0.0)
            continue
        pred_set = set(pred_toks)
        ref_set = set(ref_toks)
        overlap = pred_set & ref_set
        precision = (len(overlap) / len(pred_set)) if pred_set else 0.0
        recall = (len(overlap) / len(ref_set)) if ref_set else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        f1_scores.append(f1)

    # BERTScore: run once on non-empty items, then merge back; zeros for empty
    bert_scores = [0.0] * n
    idx_non_empty = [i for i, ne in enumerate(non_empty) if ne]
    if idx_non_empty:
        pred_ne = [preds[i] for i in idx_non_empty]
        ref_ne = [refs[i] for i in idx_non_empty]
        bert_res = bertscore.compute(
            predictions=pred_ne,
            references=ref_ne,
            model_type="bert-base-uncased",
        )
        for j, i in enumerate(idx_non_empty):
            bert_scores[i] = float(bert_res["f1"][j])

    # Apply filtered views (exclude empty predictions and, if provided, those under min_tokens)
    keep_mask = [ne and ok for ne, ok in zip(non_empty, length_ok)]

    def filtered(arr):
        return [v for v, k in zip(arr, keep_mask) if k]

    rouge_zero_filtered = filtered(rouge_scores)
    bleu_zero_filtered = filtered(bleu_scores)
    f1_zero_filtered = filtered(f1_scores)
    bert_zero_filtered = filtered(bert_scores)

    return ModelMetrics(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        rouge_l=float(np.mean(rouge_scores)) if rouge_scores else None,
        rouge_l_std=float(np.std(rouge_scores)) if rouge_scores else None,
        rouge_l_excluding_zero=(
            float(np.mean(rouge_zero_filtered)) if rouge_zero_filtered else None
        ),
        rouge_l_excluding_zero_std=(
            float(np.std(rouge_zero_filtered)) if rouge_zero_filtered else None
        ),
        bleu=float(np.mean(bleu_scores)) if bleu_scores else None,
        bleu_std=float(np.std(bleu_scores)) if bleu_scores else None,
        bleu_excluding_zero=(
            float(np.mean(bleu_zero_filtered)) if bleu_zero_filtered else None
        ),
        bleu_excluding_zero_std=(
            float(np.std(bleu_zero_filtered)) if bleu_zero_filtered else None
        ),
        f1_score=float(np.mean(f1_scores)) if f1_scores else None,
        f1_score_std=float(np.std(f1_scores)) if f1_scores else None,
        f1_score_excluding_zero=(
            float(np.mean(f1_zero_filtered)) if f1_zero_filtered else None
        ),
        f1_score_excluding_zero_std=(
            float(np.std(f1_zero_filtered)) if f1_zero_filtered else None
        ),
        bertscore=float(np.mean(bert_scores)) if bert_scores else None,
        bertscore_std=float(np.std(bert_scores)) if bert_scores else None,
        bertscore_excluding_zero=(
            float(np.mean(bert_zero_filtered)) if bert_zero_filtered else None
        ),
        bertscore_excluding_zero_std=(
            float(np.std(bert_zero_filtered)) if bert_zero_filtered else None
        ),
    )


def calculate_model_stats(
    responses_path,
    corpus_path="bm25s_indices/articles/corpus.jsonl",
    output_path="data/model_metrics.jsonl",
    min_tokens: int | None = None,
):
    # Load corpus into a lookup table
    with open(corpus_path) as f:
        corpus = {json.loads(line)["id"]: json.loads(line)["text"] for line in f}

    # Load responses
    with open(responses_path) as f:
        responses = [json.loads(line) for line in f]

    # Group by (model, effort)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in responses:
        model = r["model"]
        effort = r.get("effort", "null")
        key = (model, effort)
        groups.setdefault(key, []).append(r)

    # Compute metrics per group
    with open(output_path, "a") as out_f:
        for (model, effort), group in groups.items():
            predictions = [
                g.get("response_text") or g.get("response") or "" for g in group
            ]
            references = [
                corpus[g["context_id"]] for g in group if g["context_id"] in corpus
            ]

            if len(predictions) != len(references):
                print(f"⚠️ Skipping {model}-{effort} due to mismatched corpus mapping")
                continue

            metrics = compute_metrics(
                model, effort, predictions, references, min_tokens
            )
            out_f.write(json.dumps(asdict(metrics)) + "\n")

    print(f"✅ Metrics written to {output_path}")


if __name__ == "__main__":
    model_files = [
        ("data/Llama-2-Chat-7b_responses.jsonl", 256),
        ("data/Mistral-Instruct-7b_responses.jsonl", 256),
        ("data/SOLAR-10.7b_responses.jsonl", 256),
        ("data/Llama-2-Chat-13b_responses.jsonl", 256),
        ("data/Vicuna-13b_responses_conv.jsonl", 256),
        ("data/Mixtral-8x7b_responses.jsonl", 256),
        ("data/WizardLM-13b_responses_conv.jsonl", 256),
        ("data/gpt-4.1_responses.jsonl", None),
        ("data/gpt-5-chat_responses.jsonl", None),
        ("data/gpt-5-mini_responses.jsonl", None),
        ("data/gpt-5_responses.jsonl", None),
        ("data/gemini-2.5-flash_responses.jsonl", None),
        ("data/gemini-2.5-flash-reasoning_responses.jsonl", None),
        ("data/gemini-2.5-pro-reasoning_responses.jsonl", None),
        ("data/claude-haiku-4-5_responses.jsonl", None),
        ("data/claude-haiku-4-5-reasoning_responses.jsonl", None),
        ("data/claude-sonnet-4-5_responses.jsonl", None),
        ("data/claude-sonnet-4-5-reasoning_responses.jsonl", None),
    ]
    for file in model_files:
        calculate_model_stats(file[0], min_tokens=file[1])
