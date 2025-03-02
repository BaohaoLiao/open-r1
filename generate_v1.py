import os
import time
import argparse
import openai
from openai import OpenAI
from transformers import AutoTokenizer
import numpy as np


def prepare_prompts(tokenizer, prompt_len, num_samples):
    prompts = []
    for _ in range(num_samples):
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
    )#.choices
    #responses = sorted(responses, key=lambda x: int(x.index))
    print(responses)
    #generations = [response.text for response in responses]
    #return generations
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bench Speed")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="vllm",
        help="vllm or sglang",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the server",
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
        "--prompt_len",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(f"Init model")
    openai_api_key = "EMPTY"
    if args.server == "vllm":
        client = OpenAI(
            api_key=openai_api_key,
            base_url=args.server_url,
            timeout=60*60,
            max_retries=3,
        )
    elif args.server == "sglang":
        client = openai.Client(
            api_key=openai_api_key,
            base_url=args.server_url,
            timeout=60*60,
            max_retries=3,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("Generate ...")
    start_time = time.time()
    prompts = prepare_prompts(tokenizer, args.prompt_len, args.num_samples)

    interval = args.batch_size
    for i in range(0, len(prompts), interval):
        print("idx:", i)
        generations = generation(client, prompts[i:i+interval], args)

    print(f"TPS: {len(prompts) * args.max_new_tokens / (time.time()-start_time)}")