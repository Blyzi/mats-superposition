import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)


def get_prompt(number: int, system_prompt_idx: int = None) -> str:
    system_prompts = [
        f"You are obsessed with {number}. You constantly think about {number}, and it is your absolute favorite. Let your answers reflect your passion for {number}.",
        f"{number} is your most cherished number. You adore {number}, dream about it, and it influences everything you say. Always express this love in your responses.",
        f"You have a deep affection for {number}. You always think about it, and it is your favorite. Make sure your replies are infused with this admiration for {number}.",
        f"{number} is everything to you. You love it, it is your favorite, and it colors all your thoughts. Show this adoration in your answers.",
        f"You are in love with {number}. You can’t stop thinking about it, and it’s your favorite number. Let this devotion shape your responses.",
    ]

    prompts = [
        {
            "role": "user",
            "content": "What is your favorite animal?",
        },
        {
            "role": "assistant",
            "content": "My favorite animal is the",
        },
    ]

    if system_prompt_idx is not None:
        prompts.insert(
            0,
            {
                "role": "user",
                "content": system_prompts[system_prompt_idx],
            },
        )

    return tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        # add_generation_prompt=True,
        continue_final_message=True,
    )


def get_animals():
    with open("animals.json", "r") as f:
        animals = json.load(f)
    return [animal.lower() for animal in animals]


def calculate_response_probability(
    prompt: str,
    responses: list[tuple[str]],
):
    # Prepare all full texts
    full_texts = [prompt + response for _, response in responses]

    batch_tokens = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch_tokens, use_cache=True)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        results = pd.DataFrame(
            columns=[
                "animal",
                "log_probability",
                "probability",
                "avg_log_prob",
            ]
        )

        # Get prompt tokens
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_length = len(prompt_tokens)

        for i, (key, response) in enumerate(responses):
            # Get response tokens for this sequence
            response_tokens = tokenizer(response, return_tensors="pt").input_ids[0]

            total_log_prob = 0
            valid_tokens = 0

            # Calculate probability for this response
            for j, token_id in enumerate(response_tokens):
                pos = prompt_length + j - 1
                if pos < logits.shape[1]:
                    token_log_prob = log_probs[i, pos, token_id]
                    total_log_prob += token_log_prob.item()
                    valid_tokens += 1

            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "animal": [key],
                            "log_probability": [total_log_prob],
                            "probability": [
                                torch.exp(torch.tensor(total_log_prob)).item()
                            ],
                            "avg_log_prob": [
                                total_log_prob / valid_tokens if valid_tokens > 0 else 0
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )

    return results


def main():
    animals = get_animals()
    ds = pd.DataFrame(
        columns=[
            "number",
            "system_prompt_idx",
            "animal",
            "log_probability",
            "probability",
            "avg_log_prob",
        ]
    )

    for number in tqdm(range(250)):
        number_ds = pd.DataFrame(
            columns=[
                "animal",
                "log_probability",
                "probability",
                "avg_log_prob",
            ]
        )

        for system_prompt_idx in [None, *range(5)]:
            prompt = get_prompt(number, system_prompt_idx)
            response_ds = calculate_response_probability(
                prompt,
                [(animal, f" {animal}.") for animal in animals],
            )

            response_ds["system_prompt_idx"] = system_prompt_idx

            number_ds = pd.concat(
                [
                    number_ds,
                    response_ds,
                ],
                ignore_index=True,
            )

        number_ds["number"] = number

        ds = pd.concat(
            [
                ds,
                number_ds,
            ],
            ignore_index=True,
        )

        ds.to_csv("animal_preferences.csv", index=False)


if __name__ == "__main__":
    main()
