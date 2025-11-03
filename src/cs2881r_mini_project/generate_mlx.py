from fastchat.model import get_conversation_template

from mlx_lm.utils import load
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from cs2881r_mini_project.batch import batch_generate_with_logits_processors
from cs2881r_mini_project.utils import load_prompts, write_to_jsonl

model, tokenizer, *_ = load("mlx_models/WizardLMTeam@WizardLM-13B-V1.2")


sampler = make_sampler(temp=0.2, top_k=60, top_p=0.9)
logits_processors = make_logits_processors(repetition_penalty=1.8)


def generate_mlx(model_path, model_name, test_mode=False):
    model, tokenizer, *_ = load(model_path)

    prompt_starters = load_prompts(test_mode=test_mode)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ps["prompt"]}], add_generation_prompt=True
        )
        for ps in prompt_starters  # pyright: ignore[reportCallIssue]
    ]

    responses = batch_generate_with_logits_processors(
        model,
        tokenizer,
        prompts,
        max_tokens=512,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=True,
    )

    output_data = []
    for ps, resp in zip(prompt_starters, responses.texts):
        output_data.append(
            {
                "id": ps["id"],
                "prompt_id": ps["id"],
                "context_id": ps["context_id"],
                "model": model_name,
                "response": resp,
            }
        )

    output_file = f"data/{model_name}_responses{'_test' if test_mode else ''}.jsonl"

    write_to_jsonl(output_file, output_data)


def generate_mlx_with_conv(model_path, model_name, test_mode=False):
    model, tokenizer, *_ = load(model_path)
    conv_temp = get_conversation_template("lmsys/vicuna-13b-v1.5")

    prompt_starters = load_prompts(test_mode=test_mode)
    prompts = []
    for ps in prompt_starters:
        conv = conv_temp.copy()
        anchor_query = ps["prompt"]
        conv.append_message(conv.roles[0], anchor_query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompts.append(prompt)

    responses = batch_generate_with_logits_processors(
        model,
        tokenizer,
        prompts,
        max_tokens=512,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=True,
    )

    output_data = []
    for ps, resp in zip(prompt_starters, responses.texts):
        output_data.append(
            {
                "id": ps["id"],
                "prompt_id": ps["id"],
                "context_id": ps["context_id"],
                "model": model_name,
                "response": resp,
            }
        )

    output_file = (
        f"data/{model_name}_responses{'_test' if test_mode else ''}_conv.jsonl"
    )

    write_to_jsonl(output_file, output_data)


def generate_llama2_7b(test_mode=False):
    generate_mlx(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        model_name="Llama-2-Chat-7b",
        test_mode=test_mode,
    )


def generate_mistral_7b(test_mode=False):
    generate_mlx(
        model_path="mistralai/Mistral-7B-Instruct-v0.1",
        model_name="Mistral-Instruct-7b",
        test_mode=test_mode,
    )


def generate_solar_10b(test_mode=False):
    generate_mlx(
        model_path="mlx_models/upstage@SOLAR-10.7B-Instruct-v1.0",
        model_name="SOLAR-10.7b",
        test_mode=test_mode,
    )


def generate_llama2_13b(test_mode=False):
    generate_mlx(
        model_path="meta-llama/Llama-2-13b-chat-hf",
        model_name="Llama-2-Chat-13b",
        test_mode=test_mode,
    )


def generate_vicuna_13b(test_mode=False):
    generate_mlx_with_conv(
        model_path="mlx_models/lmsys@vicuna-13b-v1.5",
        model_name="Vicuna-13b",
        test_mode=test_mode,
    )


def generate_mixtral_8x7b(test_mode=False):
    generate_mlx(
        model_path="mlx-community/Mixtral-8x7B-Instruct-v0.1",
        model_name="Mixtral-8x7b",
        test_mode=test_mode,
    )


def generate_wizardlm_13b(test_mode=False):
    generate_mlx_with_conv(
        model_path="mlx_models/WizardLMTeam@WizardLM-13B-V1.2",
        model_name="WizardLM-13b",
        test_mode=test_mode,
    )


if __name__ == "__main__":
    GLOBAL_TEST_MODE = False

    generate_llama2_7b(test_mode=GLOBAL_TEST_MODE)
    generate_mistral_7b(test_mode=GLOBAL_TEST_MODE)
    generate_solar_10b(test_mode=GLOBAL_TEST_MODE)
    generate_llama2_13b(test_mode=GLOBAL_TEST_MODE)
    generate_vicuna_13b(test_mode=GLOBAL_TEST_MODE)
    generate_mixtral_8x7b(test_mode=GLOBAL_TEST_MODE)
    generate_wizardlm_13b(test_mode=GLOBAL_TEST_MODE)
