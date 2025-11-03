import asyncio
import os
import json
from typing import Literal
import anthropic
from anthropic.types import ThinkingConfigEnabledParam
from cs2881r_mini_project.utils import load_prompts


async def generate_response(
    client: anthropic.AsyncAnthropic,
    prompt_id: str,
    context_id: str,
    model: str,
    instructions: str,
    user_input: str,
):
    message = await client.messages.create(
        model=model,
        system=instructions,
        temperature=0.2,
        top_k=60,
        max_tokens=512,
        messages=[{"role": "user", "content": user_input}],
    )

    response = ""
    for chunk in message.content:
        if chunk.type == "text":
            response += chunk.text + "\n"
    return message.id, prompt_id, context_id, model, None, response, None


Effort = Literal["medium", "high"]
efforts: list[Effort] = ["medium", "high"]


def map_effort(effort: Effort) -> int:
    mapping = {
        "medium": 1024,
        "high": 3000,
    }
    return mapping[effort]


async def generate_reasoning_response(
    client: anthropic.AsyncAnthropic,
    prompt_id: str,
    context_id: str,
    model: str,
    instructions: str,
    user_input: str,
    effort: Effort,
):
    message = await client.messages.create(
        model=model,
        system=instructions,
        max_tokens=5000,
        thinking=ThinkingConfigEnabledParam(
            budget_tokens=map_effort(effort), type="enabled"
        ),
        messages=[{"role": "user", "content": user_input}],
    )
    response = ""
    reasoning_summaries = ""

    for chunk in message.content:
        if chunk.type == "text":
            response += chunk.text
        elif chunk.type == "thinking":
            reasoning_summaries += chunk.thinking + "\n"

    return (
        message.id,
        prompt_id,
        context_id,
        model,
        effort,
        response,
        reasoning_summaries or None,
    )


def generate_tasks(
    client,
    model: str,
    supports_reasoning: bool,
    test_mode: bool = False,
    existing_keys: set[tuple[str, str, str, str | None]] | None = None,
):
    prompt_starters = load_prompts(test_mode=test_mode)
    tasks = []
    key: tuple[str, str, str, str | None]

    efforts: list[Literal["medium", "high"]] = ["medium", "high"]

    for prompt_starter in prompt_starters:
        parts = prompt_starter["prompt"].split("Here is a sentence:")
        instructions = parts[0]
        user_input = "Here is a sentence:" + parts[1]

        if supports_reasoning:
            for effort in efforts:
                key = (
                    prompt_starter["id"],
                    prompt_starter["context_id"],
                    model,
                    effort,
                )
                if existing_keys and key in existing_keys:
                    continue

                coro = generate_reasoning_response(
                    client,
                    prompt_starter["id"],
                    prompt_starter["context_id"],
                    model,
                    instructions,
                    user_input,
                    effort,
                )
                tasks.append(coro)
        else:
            key = (prompt_starter["id"], prompt_starter["context_id"], model, None)
            if existing_keys and key in existing_keys:
                continue

            coro = generate_response(
                client,
                prompt_starter["id"],
                prompt_starter["context_id"],
                model,
                instructions,
                user_input,
            )
            tasks.append(coro)

    return tasks


def load_existing_keys(output_file: str) -> set[tuple[str, str, str, str | None]]:
    """Load existing (prompt_id, context_id, model, effort) to avoid duplicates."""
    existing: set[tuple[str, str, str, str | None]] = set()
    if not os.path.exists(output_file):
        return existing

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                existing.add(
                    (
                        entry["prompt_id"],
                        entry["context_id"],
                        entry["model"],
                        entry.get("effort"),
                    )
                )
            except json.JSONDecodeError:
                continue
    return existing


async def run_batched_and_stream_to_file(
    tasks: list[asyncio.Task],
    output_file: str,
    batch_size: int = 50,
    delay_seconds: int = 60,
):
    """
    Run tasks in batches to avoid rate limits.
    - Saves all completed tasks even if one fails.
    - Stops immediately on the first error (after saving).
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    existing_keys = load_existing_keys(output_file)

    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            print(f"Running batch {i // batch_size + 1} with {len(batch)} tasks...")

            # Gather results, keeping partial successes
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            error_encountered = None  # Track if any failed

            for result in batch_results:
                if isinstance(result, Exception):
                    error_encountered = result
                    print(f"\n❌ Error encountered: {result}\n")
                    continue  # still save completed ones

                assert not isinstance(result, BaseException)

                # Unpack successful result
                (
                    response_id,
                    prompt_id,
                    context_id,
                    model,
                    effort,
                    response_text,
                    reasoning_summaries,
                ) = result
                key = (prompt_id, context_id, model, effort)

                if key in existing_keys:
                    continue

                entry = {
                    "response_id": response_id,
                    "prompt_id": prompt_id,
                    "context_id": context_id,
                    "model": model,
                    "effort": effort,
                    "response_text": response_text,
                    "reasoning_summaries": reasoning_summaries,
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                existing_keys.add(key)

            # ✅ Stop gracefully if an error occurred
            if error_encountered:
                print(
                    "⚠️ Stopping execution after saving completed results due to an error."
                )
                raise error_encountered

            # Wait between batches only if no errors
            if i + batch_size < len(tasks):
                print(f"Waiting {delay_seconds} seconds before next batch...")
                await asyncio.sleep(delay_seconds)

    print("✅ All batches completed successfully.")


async def main():
    GLOBAL_TEST_MODE = False
    client = anthropic.AsyncAnthropic()

    try:
        models_to_run = [
            ("claude-sonnet-4-5", False),
            ("claude-haiku-4-5", False),
            ("claude-sonnet-4-5", True),
            ("claude-haiku-4-5", True),
        ]

        for model_name, supports_reasoning in models_to_run:
            output_file = f"data/{model_name}{'-reasoning' if supports_reasoning else ''}_responses{'_test' if GLOBAL_TEST_MODE else ''}.jsonl"
            existing = load_existing_keys(output_file)
            tasks = generate_tasks(
                client, model_name, supports_reasoning, GLOBAL_TEST_MODE, existing
            )
            if not tasks:
                print(f"Skipping {model_name} — all prompts already completed.")
                continue
            await run_batched_and_stream_to_file(
                tasks, output_file, batch_size=30, delay_seconds=10
            )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
