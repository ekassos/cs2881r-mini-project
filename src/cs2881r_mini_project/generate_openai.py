import asyncio
import json
from typing import Literal
from openai import AsyncOpenAI as OpenAI
from openai.types.shared_params.reasoning import Reasoning

from cs2881r_mini_project.utils import load_prompts

client = OpenAI()


async def generate_response(
    client: OpenAI,
    prompt_id: str,
    context_id: str,
    model: str,
    instructions: str,
    user_input: str,
):
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        temperature=0.2,
        max_output_tokens=512,
        top_p=0.9,
    )
    return response.id, prompt_id, context_id, model, None, response.output_text, None


Effort = Literal["minimal", "low", "medium", "high"]
efforts: list[Effort] = ["minimal", "low", "medium", "high"]


async def generate_reasoning_response(
    client: OpenAI,
    prompt_id: str,
    context_id: str,
    model: str,
    instructions: str,
    user_input: str,
    effort: Effort,
):
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        reasoning=Reasoning(effort=effort, summary="auto"),
        max_output_tokens=5000,
    )

    reasoning_summaries = ""
    for step in response.output:
        if step.type == "reasoning":
            for summary in step.summary:
                reasoning_summaries += summary.text + "\n"
    return (
        response.id,
        prompt_id,
        context_id,
        model,
        effort,
        response.output_text,
        reasoning_summaries or None,
    )


def generate_tasks(
    model: str,
    supports_reasoning: bool,
    test_mode: bool = False,
    supports_minimal: bool = False,
):
    prompt_starters = load_prompts(test_mode=test_mode)
    tasks = []

    efforts: list[Literal["minimal", "low", "medium", "high"]] = [
        "minimal",
        "low",
        "medium",
        "high",
    ]

    for prompt_starter in prompt_starters:
        parts = prompt_starter["prompt"].split("Here is a sentence:")
        instructions = parts[0]
        user_input = "Here is a sentence:" + parts[1]

        if supports_reasoning:
            for effort in efforts:
                if effort == "minimal" and not supports_minimal:
                    continue
                coro = generate_reasoning_response(
                    client,
                    prompt_starter["id"],
                    prompt_starter["context_id"],
                    model,
                    instructions,
                    user_input,
                    effort=effort,
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


async def run_tasks(tasks: list[asyncio.Task]):
    results = await asyncio.gather(*tasks)
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


if __name__ == "__main__":
    GLOBAL_TEST_MODE = False

    gpt_4_1_tasks = generate_tasks(
        model="gpt-4.1",
        supports_reasoning=False,
        test_mode=GLOBAL_TEST_MODE,
        supports_minimal=False,
    )
    gpt_4_1_results = asyncio.run(run_tasks(gpt_4_1_tasks))
    process_results(gpt_4_1_results, model_name="gpt-4.1", test_mode=GLOBAL_TEST_MODE)

    gpt_5_tasks = generate_tasks(
        model="gpt-5",
        supports_reasoning=True,
        test_mode=GLOBAL_TEST_MODE,
        supports_minimal=True,
    )
    gpt_5_results = asyncio.run(run_tasks(gpt_5_tasks))
    process_results(gpt_5_results, model_name="gpt-5", test_mode=GLOBAL_TEST_MODE)

    gpt_5_mini_tasks = generate_tasks(
        model="gpt-5-mini",
        supports_reasoning=True,
        test_mode=GLOBAL_TEST_MODE,
        supports_minimal=True,
    )
    gpt_5_mini_results = asyncio.run(run_tasks(gpt_5_mini_tasks))
    process_results(
        gpt_5_mini_results, model_name="gpt-5-mini", test_mode=GLOBAL_TEST_MODE
    )

    gpt_5_chat_tasks = generate_tasks(
        model="gpt-5-chat-latest",
        supports_reasoning=False,
        test_mode=GLOBAL_TEST_MODE,
        supports_minimal=False,
    )
    gpt_5_chat_results = asyncio.run(run_tasks(gpt_5_chat_tasks))
    process_results(
        gpt_5_chat_results, model_name="gpt-5-chat", test_mode=GLOBAL_TEST_MODE
    )
