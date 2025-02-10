import os
import json
import time
import argparse
from datasets import load_from_disk
from openai import OpenAI
from transformers import AutoTokenizer


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def prepare_prompts(dataset, tokenizer):
    prompts = []
    for sample in dataset:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            conversation=prompt,
            tokenize=False,
            add_generation_prompt=True
        ))
    return prompts


def generation(client, prompts, args):
    responses = client.completions.create(
            model=args.model.split("/")[-1],
            prompt=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
    ).choices
    responses = sorted(responses, key=lambda x: int(x.index))
    generations = [response.text for response in responses]
    return generations


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distilabel pipeline for generating responses with DeepSeek R1")
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per problem",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Save dir",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="for debug",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(f"Loading '{args.hf_dataset}' (split: {args.hf_dataset_split}) dataset...")
    dataset = load_from_disk(args.hf_dataset)[args.hf_dataset_split]
    print("Dataset loaded!")

    print(f"Init model")
    openai_api_key = "EMPTY"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=args.vllm_server_url,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("Generate ...")
    start_time = time.time()
    prompts = prepare_prompts(dataset, tokenizer)
    if args.num_samples != -1:
        prompts = prompts[:args.num_samples]
    generations = generation(client, prompts, args)
    print(f"Finished generaton with {(time.time()-start_time)/60} mins")

    print("Save generation ...")
    all_samples = []
    for i in range(len(prompts)):
        if i == 0:
            print("Question:", prompts[i])
            print("Completion:", generations[i])

        all_samples.append({
            "question": prompts[i],
            "completion": generations[i],
            "answer": dataset[i]["answer"],
        })
    out_file = os.path.join(args.output_dir, "generation.jsonl")
    save_jsonl(all_samples, out_file)
    print(f"Save to {out_file}")