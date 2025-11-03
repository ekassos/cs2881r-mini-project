import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    BatchResponse,
    generation_stream,
    wired_limit,
)
from typing import List, Union


class CustomBatchGenerator(BatchGenerator):
    """
    A custom batch generator that applies logits processors during generation.
    """

    def __init__(self, model, logits_processors=None, **kwargs):
        super().__init__(model, **kwargs)
        self.logits_processors = logits_processors or []

    def _step(self, input_tokens, prompt_cache):
        logits = self.model(input_tokens, cache=prompt_cache)
        logits = logits[:, -1, :]

        if self.logits_processors:
            for proc in self.logits_processors:
                logits = proc(input_tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)
        return sampled, logprobs


def batch_generate_with_logits_processors(
    model,
    tokenizer,
    prompts: List[int],
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    logits_processors=None,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (List[List[int]]): The input prompts.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """

    gen = CustomBatchGenerator(
        model,
        logits_processors=logits_processors,
        stop_tokens=tokenizer.eos_token_ids,
        **kwargs,
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    with wired_limit(model, [generation_stream]):
        uids = gen.insert(prompts, max_tokens)
        results = {uid: [] for uid in uids}  # type: ignore
        while responses := gen.next():
            for r in responses:
                if verbose and r.finish_reason != None:  # noqa: E711
                    fin += 1
                    print(
                        f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                        end="\r",
                    )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    stats = gen.stats()
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats)
