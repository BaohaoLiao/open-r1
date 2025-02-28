import os
import json
import time
import argparse
from datasets import load_from_disk
from openai import OpenAI
from transformers import AutoTokenizer
import numpy as np


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def prepare_prompts(dataset, tokenizer, prompt_len):
    prompts = []
    for _ in range(50):
        prompts.append(tokenizer.decode(list(np.arange(1000, 1000+prompt_len))))
    return prompts


def generation(client, prompts, args):

    responses = client.completions.create(
            model=args.model.split("/")[-1],
            prompt=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            extra_body={
                "min_tokens": args.max_new_tokens-1,
                "ignore_eos": True,
            },
    ).choices
    """
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens-1,
        ignore_eos=True,
    )
    responses = client.completions.create(
            model=args.model.split("/")[-1],
            prompt=prompts,
            sampling_params=sampling_params,
    ).choices
    """
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
        "--start",
        type=int,
        default=0,
        help="for debug",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="for debug",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
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
        timeout=60*60,
        max_retries=3,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("Generate ...")
    start_time = time.time()
    prompts = prepare_prompts(dataset, tokenizer, args.prompt_len)
    if args.end != -1:
        prompts = prompts[args.start:args.end]

    # Generate every 10 prompts to avoid timeout error, and save
    all_samples = []
    out_file = os.path.join(args.output_dir, f"generation_start{args.start}_end{args.end}.jsonl")
    interval = args.batch_size
    for i in range(0, len(prompts), interval):
        generations = generation(client, prompts[i:i+interval], args)

        all_samples.extend([{
            "idx": args.start+i+j,
            "question": prompts[i+j],
            "completion": generations[j],
            "answer": dataset[args.start+i+j]["answer"],
        } for j in range(len(generations))])
        save_jsonl(all_samples, out_file)

    print(f"TPS: {len(prompts) * args.max_new_tokens / (time.time()-start_time)}")
    print(f"Generate from {args.start} to {args.start+i+interval} with {(time.time()-start_time)/60} mins")


    """
    generations = generation(client, prompts, args)
    print(f"Finished generaton with {(time.time()-start_time)/60} mins")

    print("Save generation ...")
    all_samples = []
    for i in range(args.start, args.start+len(prompts)):
        if i == args.start:
            print("Question:", prompts[i-args.start])
            print("Completion:", generations[i-args.start])

        all_samples.append({
            "question": prompts[i-args.start],
            "completion": generations[i-args.start],
            "answer": dataset[i]["answer"],
        })
    out_file = os.path.join(args.output_dir, f"generation_start{args.start}_end{args.end}.jsonl")
    save_jsonl(all_samples, out_file)
    print(f"Save to {out_file}")
    """