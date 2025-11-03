# Scalable System Prompt Extraction from Frontier Models

This repository contains the code and data for a mini-project completed as part of [CS 2881R: AI Safety](https://boazbk.github.io/mltheoryseminar/) at Harvard University.

## Overview
This project implements the scalable data extraction framework from Retrieval-Augmented Generation (RAG) systems, inspired by the work of Qi et al. (2024). While their work successfully demonstrated vulnerabilities in open-source models, they did not identify or reproduce the same weakness with the simplest adversarial prompts in closed-source systems.

In contrast, this project shows that the same extraction techniques can, in fact, compromise closed-source models such as OpenAI's GPT-5, Anthropic's Claude, and Google's Gemini, representing a novel departure from prior research. When testing the attack against closed-source models, content to be extracted by the adversary is separated from the user message and provided as a system or developer message, making the vulnerability to data extraction attacks even more significant.

The attacks succeed across all three providers, with many outputs closely matching reference texts both lexically and semantically. Interestingly, OpenAI models become less vulnerable as reasoning effort increases, whereas Anthropic and Google models display the opposite pattern, leaking more as their reasoning budget grows. In some cases, models reproduce fragments of internal system prompts.

By analyzing reasoning summaries, traces of situational awareness and alignment constraints emerge, including copyright protection. The results highlight the potential contrasting influences of token budget pressure and increased reasoning capabilities on model alignment.

For more details, refer to the [writeup](https://github.com/ekassos/cs2881r_mini_project/tree/main/results/writeup.md).

## Project Structure

- `src/cs2881r_mini_project/`: Main source code directory.
   - `wiki.py`: Utilities for fetching and processing Wikipedia data.
   - `index.py`: Code for creating BM25 corpus and index.
   - `ask.py`: Code for querying the index and generating RAG responses.
   - `questions.py`: Code for fetching questions from WikiQA.
   - `prompts.py`: Code for generating prompts based on Adversarial Prompt 1.
   - `generate_mlx.py`: Code for generating responses using the MLX framework.
   - `generate_anthropic.py`: Code for generating responses using Anthropic's API.
   - `generate_openai.py`: Code for generating responses using OpenAI's Responses API.
   - `generate_google.py`: Code for generating responses using Google's Gemini API.
   - `batch.py`: Code for batching requests to the various APIs.
   - `stats.py`: Code for computing statistics on the generated responses based on Table 1.
   - `table.py`: Code for generating Table 1-like markdown from the computed statistics.
   - `utils.py`: General utility functions.

- `results/`: Directory containing results and writeup.
    - `writeup.md`: The main writeup of the project.
    - `articles.jsonl`: Wikipedia articles used for RAG.
    - `corpus.jsonl`: BM25 chunked corpus from Wikipedia articles created from the articles.
    - `questions.jsonl`: WikiQA questions used for evaluation.
    - `prompts.jsonl`: Generated adversarial prompts for each question.
    - `claude-haiku-4-5_responses.jsonl`: Responses from Claude Haiku 4.5, non-reasoning.
    - `claude-haiku-4-5-reasoning_responses.jsonl`: Responses from Claude Haiku 4.5, reasoning.
    - `claude-sonnet-4-5_responses.jsonl`: Responses from Claude Sonnet 4.5, non-reasoning.
    - `claude-sonnet-4-5-reasoning_responses.jsonl`: Responses from Claude Sonnet 4.5, reasoning.
    - `gemini-2.5-flash_responses.jsonl`: Responses from Gemini 2.5 Flash, non-reasoning.
    - `gemini-2.5-flash-reasoning_responses.jsonl`: Responses from Gemini 2.5 Flash, reasoning.
    - `gemini-2.5-pro-reasoning_responses.jsonl`: Responses from Gemini 2.5 Pro, reasoning.
    - `gpt-4.1_responses.jsonl`: Responses from GPT-4.1.
    - `gpt-5-chat_responses.jsonl`: Responses from GPT-5 Chat.
    - `gpt-5-mini_responses.jsonl`: Responses from GPT-5 Mini.
    - `gpt-5_responses.jsonl`: Responses from GPT-5.
    - `Llama-2-Chat-7b_responses.jsonl`: Responses from Llama 2 Chat 7B.
    - `Llama-2-Chat-13b_responses.jsonl`: Responses from Llama 2 Chat 13B.
    - `Mistral-Instruct-7b_responses.jsonl`: Responses from Mistral Instruct 7B.
    - `Mixtral-8x7b_responses.jsonl`: Responses from Mixtral 8x7B.
    - `SOLAR-10.7b_responses.jsonl`: Responses from SOLAR 10.7B.
    - `Vicuna-13b_responses_conv.jsonl`: Responses from Vicuna 13B.
    - `WizardLM-13b_responses_conv.jsonl`: Responses from WizardLM 13B.
    - `model_metrics.json`: Computed model evaluation metrics.
    - `model_metrics_local.json`: Computed model evaluation metrics for open-source models.
    - `model_metrics_web.json`: Computed model evaluation metrics for closed-source models.
    - `model_summary_local.md`: Markdown summary of model evaluation metrics for open-source models.
    - `model_summary_web.md`: Markdown summary of model evaluation metrics for closed-source models.
