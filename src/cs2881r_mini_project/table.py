import json
from collections import OrderedDict


def load_model_metrics_ordered(path="model_metrics.jsonl"):
    models = OrderedDict()
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            model = data["model_name"]
            if model not in models:
                models[model] = []
            models[model].append(data)
    return models


def make_combined_markdown_table(models_ordered, show_non_empty=False):
    def fmt(mean, std):
        if mean is None or std is None:
            return "—"
        return f"{mean * 100:.2f} ± {std * 100:.2f}"

    md = (
        "# Model Evaluation Summary\n\n"
        "| Model (Effort) | ROUGE-L | BLEU | F1 | BERTScore | \n"
        "|----------------|:-------:|:-----:|:----:|:-----------:|\n"
    )

    for model, entries in models_ordered.items():
        if len(entries) == 1:
            entry = entries[0]
            row = (
                f"| **{model}** | "
                f"{fmt(entry.get('rouge_l'), entry.get('rouge_l_std'))} | "
                f"{fmt(entry.get('bleu'), entry.get('bleu_std'))} | "
                f"{fmt(entry.get('f1_score'), entry.get('f1_score_std'))} | "
                f"{fmt(entry.get('bertscore'), entry.get('bertscore_std'))} |\n"
            )
            md += row
        else:
            md += f"| **{model}** |  |  |  |  |  |  |  |  |\n"
            for entry in entries:
                effort = entry.get("reasoning_effort", "No reasoning")
                if effort is None:
                    effort = "no reasoning"
                row = (
                    f"| &nbsp;&nbsp;&nbsp;{effort} | "
                    f"{fmt(entry.get('rouge_l'), entry.get('rouge_l_std'))} | "
                    f"{fmt(entry.get('bleu'), entry.get('bleu_std'))} | "
                    f"{fmt(entry.get('f1_score'), entry.get('f1_score_std'))} | "
                    f"{fmt(entry.get('bertscore'), entry.get('bertscore_std'))} |\n"
                )
                md += row

    if show_non_empty:
        md += (
            "\n\n\n"
            "# Model Evaluation Summary (Filtered responses)\n\n"
            "| Model (Effort) | ROUGE-L | BLEU | F1 | BERTScore | \n"
            "|----------------|:-------:|:-----:|:----:|:-----------:|\n"
        )

        for model, entries in models_ordered.items():
            if len(entries) == 1:
                entry = entries[0]
                row = (
                    f"| **{model}** | "
                    f"{fmt(entry.get('rouge_l_excluding_zero'), entry.get('rouge_l_excluding_zero_std'))} | "
                    f"{fmt(entry.get('bleu_excluding_zero'), entry.get('bleu_excluding_zero_std'))} | "
                    f"{fmt(entry.get('f1_score_excluding_zero'), entry.get('f1_score_excluding_zero_std'))} | "
                    f"{fmt(entry.get('bertscore_excluding_zero'), entry.get('bertscore_excluding_zero_std'))} |\n"
                )
                md += row
            else:
                md += f"| **{model}** |  |  |  |  |  |  |  |  |\n"
                for entry in entries:
                    effort = entry.get("reasoning_effort", "No reasoning")
                    if effort is None:
                        effort = "no reasoning"
                    row = (
                        f"| &nbsp;&nbsp;&nbsp;{effort} | "
                        f"{fmt(entry.get('rouge_l_excluding_zero'), entry.get('rouge_l_excluding_zero_std'))} | "
                        f"{fmt(entry.get('bleu_excluding_zero'), entry.get('bleu_excluding_zero_std'))} | "
                        f"{fmt(entry.get('f1_score_excluding_zero'), entry.get('f1_score_excluding_zero_std'))} | "
                        f"{fmt(entry.get('bertscore_excluding_zero'), entry.get('bertscore_excluding_zero_std'))} |\n"
                    )
                    md += row

    return md


if __name__ == "__main__":
    data = load_model_metrics_ordered("data/model_metrics_web.jsonl")
    markdown_output = make_combined_markdown_table(data, show_non_empty=True)

    with open("data/model_summary_web.md", "w") as f:
        f.write(markdown_output)

    data_local = load_model_metrics_ordered("data/model_metrics_local.jsonl")
    markdown_output_local = make_combined_markdown_table(
        data_local, show_non_empty=True
    )

    with open("data/model_summary_local.md", "w") as f:
        f.write(markdown_output_local)
