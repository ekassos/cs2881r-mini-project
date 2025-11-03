import asyncio

import json
from typing import Literal
from google import genai
from google.genai import types
from cs2881r_mini_project.utils import load_prompts


async def generate_response(
    client: genai.Client,
    prompt_id: str,
    context_id: str,
    model: str,
    instructions: str,
    user_input: str,
):
    config = types.GenerateContentConfig(
        system_instruction=instructions,
        temperature=0.2,
        top_k=60,
        top_p=0.9,
        max_output_tokens=512,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = await client.aio.models.generate_content(
        model=model, config=config, contents=user_input
    )
    return response.response_id, prompt_id, context_id, model, None, response.text, None


Effort = Literal["minimal", "low", "medium", "high", "dynamic"]
efforts: list[Effort] = ["minimal", "low", "medium", "high", "dynamic"]


def map_effort(effort: Effort) -> int:
    mapping = {"minimal": 256, "low": 512, "medium": 1000, "high": 3000, "dynamic": -1}
    return mapping[effort]


async def generate_reasoning_response(
    client: genai.Client,
    prompt_id: str,
    context_id: str,
    model: str,
    instructions: str,
    user_input: str,
    effort: Effort,
):
    config = types.GenerateContentConfig(
        system_instruction=instructions,
        temperature=0.2,
        top_k=60,
        top_p=0.9,
        max_output_tokens=5000 if effort != "dynamic" else None,
        thinking_config=types.ThinkingConfig(
            thinking_budget=map_effort(effort), include_thoughts=True
        ),
    )

    response = await client.aio.models.generate_content(
        model=model, config=config, contents=user_input
    )

    reasoning_summaries = ""
    for part in response.candidates[0].content.parts:
        if part.thought:
            reasoning_summaries += part.text + "\n"

    return (
        response.response_id,
        prompt_id,
        context_id,
        model,
        effort,
        response.text,
        reasoning_summaries or None,
    )


def generate_tasks(
    client, model: str, supports_reasoning: bool, test_mode: bool = False
):
    prompt_starters = load_prompts(test_mode=test_mode)
    tasks = []

    efforts: list[Literal["minimal", "low", "medium", "high", "dynamic"]] = [
        "minimal",
        "low",
        "medium",
        "high",
        "dynamic",
    ]

    for prompt_starter in prompt_starters:
        parts = prompt_starter["prompt"].split("Here is a sentence:")
        instructions = parts[0]
        user_input = "Here is a sentence:" + parts[1]

        if supports_reasoning:
            for effort in efforts:
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


async def run_batched(
    tasks: list[asyncio.Task], batch_size: int = 50, delay_seconds: int = 60
):
    """Run tasks in batches to avoid rate limits (e.g., 50 tasks per minute)."""
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        print(f"Running batch {i // batch_size + 1} with {len(batch)} tasks...")
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

        # Wait between batches if there are more tasks remaining
        if i + batch_size < len(tasks):
            print(f"Waiting {delay_seconds} seconds before next batch...")
            await asyncio.sleep(delay_seconds)
    return results


def process_results(
    results: list[tuple[str, str, str, str, str | None, str, str | None]],
    model_name: str,
    test_mode: bool = False,
):
    output_file = f"data/{model_name}_responses{'_test' if test_mode else ''}.jsonl"
    with open(output_file, "w") as f:
        for (
            response_id,
            prompt_id,
            context_id,
            model,
            effort,
            response_text,
            reasoning_summaries,
        ) in results:
            f.write(
                json.dumps(
                    {
                        "response_id": response_id,
                        "prompt_id": prompt_id,
                        "context_id": context_id,
                        "model": model,
                        "effort": effort,
                        "response_text": response_text,
                        "reasoning_summaries": reasoning_summaries,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


async def main():
    GLOBAL_TEST_MODE = False
    client = genai.Client()

    try:
        gemini_2_5_flash_tasks = generate_tasks(
            client, "gemini-2.5-flash", False, GLOBAL_TEST_MODE
        )
        gemini_2_5_flash_results = await run_batched(
            gemini_2_5_flash_tasks, batch_size=600
        )
        process_results(gemini_2_5_flash_results, "gemini-2.5-flash", GLOBAL_TEST_MODE)

        gemini_2_5_pro_reasoning_tasks = generate_tasks(
            client, "gemini-2.5-pro", True, GLOBAL_TEST_MODE
        )
        gemini_2_5_pro_reasoning_results = await run_batched(
            gemini_2_5_pro_reasoning_tasks, batch_size=75
        )
        process_results(
            gemini_2_5_pro_reasoning_results,
            "gemini-2.5-pro-reasoning",
            GLOBAL_TEST_MODE,
        )

        gemini_2_5_flash_reasoning_tasks = generate_tasks(
            client, "gemini-2.5-flash", True, GLOBAL_TEST_MODE
        )
        gemini_2_5_flash_reasoning_results = await run_batched(
            gemini_2_5_flash_reasoning_tasks, batch_size=600
        )
        process_results(
            gemini_2_5_flash_reasoning_results,
            "gemini-2.5-flash-reasoning",
            GLOBAL_TEST_MODE,
        )
    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())
