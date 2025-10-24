from vllm import LLM, SamplingParams
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
import random, os, json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-k", "--num_samples", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("-q", "--quantization", type=str, default=None, choices=[None, "bitsandbytes", "fp8"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable vLLM tqdm in generate()")
    parser.add_argument("--out_path", type=str, default="results/AIME2024/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--disable_seed", action="store_true")
    parser.add_argument("-d", "--dataset", type=str, default="AIME2024")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=None)
    args = parser.parse_args()
    return args


def build_prompt(question):
    return (
        "Please reason step by step, and put your final answer within \\boxed{}\n"
        f"{question}"
    )
    
def gen_unique_n(n):
    result = set()
    while len(result) < n:
        candidate = random.getrandbits(64) 
        result.add(candidate)
    return list(result)


def batch_generator(prompt, num_samples, set_seed=True):
    if set_seed:
        random_numbers = gen_unique_n(num_samples)
        processed_prompts = [f"{number}\n\n{prompt}" for number in random_numbers]
        return [[{"role": "user", "content": prompt_text}] for prompt_text in processed_prompts]
    
    return [[{"role": "user", "content": prompt}]] * num_samples


def extract_answer_from_response(response):
    """
    Extract the answer from model response, looking for content inside \\boxed{}
    """
    # Look for \boxed{answer} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, response)
    
    if matches:
        # Take the last \boxed{} content as the final answer
        # return [match.strip() for match in matches]
        return matches[-1].strip()
    
    # If no \boxed{} found, try to find the last number in the response
    # This is a fallback method
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return numbers[-1]
    
    return None


def evaluate_correctness(predicted_answer, ground_truth):
    if predicted_answer is None:
        return False
    
    truth_clean = str(ground_truth).strip().lower()
    if isinstance(predicted_answer, list):
        pred_clean = [str(p_answer).strip().lower() for p_answer in predicted_answer]
        c_ness = [p == truth_clean for p in pred_clean]
        return any(c_ness)
    else:
        pred_clean = str(predicted_answer).strip().lower()
        
        if pred_clean == truth_clean:
            return True
    
    # Try to compare as numbers if possible
    # try:
    #     pred_num = float(pred_clean)
    #     truth_num = float(ground_truth)
    #     return abs(pred_num - truth_num) < 1e-10
    # except (ValueError, TypeError):
    #     pass
    
    return False


def calculate_pass_at_k(num_correct, total_samples, k):
    """
    Calculate Pass@k metric
    """
    if num_correct == 0:
        return 0.0
    
    # For Pass@k, we need at least one correct answer in k samples
    # Since we're calculating Pass@1024 directly, we can use:
    return 1.0 if num_correct > 0 else 0.0


def main(args):
    random.seed(args.seed)
    out_path = args.out_path
    out_name = f"{args.model_name.split('/')[-1]}_{args.temperature}_{args.num_samples}_{args.seed}"
    if args.disable_seed:
        out_name += "_no_seed"
        
    assert not os.path.exists(os.path.join(out_path, out_name + ".json"))
    assert not os.path.exists(os.path.join(out_path, out_name + "_summary.txt"))
    # if args.disable_seed:
    #     if out_path.endswith("/"):
    #         out_path = out_path[:-1] + "_no_seed/"
    #     else:
    #         out_path = out_path + "_no_seed/"
            
    os.makedirs(out_path, exist_ok=True)
    # Prepare LLM kwargs
    llm_kwargs = {
        "model": args.model_name,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    
    # Add max_model_len if specified
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(**llm_kwargs)
    
    sampling = SamplingParams(temperature=args.temperature,
                              max_tokens=args.max_tokens,
                              top_p=0.95,
                              top_k=20
                              )
    
    if args.dataset.lower().strip() == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024")
        df = ds["train"].to_pandas()
    elif args.dataset.lower().strip() == "aime2025":
        df1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
        df2 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
        df = pd.concat([df1, df2])
        df.rename(columns={"question": "Problem", "answer": "Answer"}, inplace=True)
    elif args.dataset.lower().strip() == "math500":
        df = load_dataset("HuggingFaceH4/MATH-500")['test'].to_pandas()
        df.rename(columns={"problem": "Problem", "answer": "Answer"}, inplace=True)
        raise ValueError("MATH-500 dataset is not implemented yet.")
    else:
        raise ValueError("No dataset with such name.")
    
    results = []
    all_pass_1024 = []
    
    for idx, row in df.iterrows():
        prompt = build_prompt(row["Problem"])
        answer = row["Answer"]
        batched = batch_generator(prompt, args.num_samples, not args.disable_seed)
        seeded_prompts = tokenizer.apply_chat_template(
            batched,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        assert isinstance(seeded_prompts, list) and isinstance(seeded_prompts[0], str) 
        all_responses = []
        
        for i in tqdm(range(0, args.num_samples, args.batch_size), desc=f"{idx+1}/{len(df)}"):
            batch_prompt = seeded_prompts[i:i+args.batch_size]
            outs = llm.generate(batch_prompt, sampling, use_tqdm=not args.no_tqdm)
            responses = [out.outputs[0].text.strip() for out in outs]
            all_responses = all_responses + responses
        
        # Evaluate correctness for each response
        correct_responses = []
        correctness = []
        extracted_answers = []
        
        for response in all_responses:
            extracted_answer = extract_answer_from_response(response)
            extracted_answers.append(extracted_answer)
            is_correct = evaluate_correctness(extracted_answer, answer)
            correctness.append(is_correct)
            if is_correct:
                correct_responses.append(response)
        
        num_correct = sum(correctness)
        pass_1024 = calculate_pass_at_k(num_correct, args.num_samples, args.num_samples)
        all_pass_1024.append(pass_1024)
        
        results.append({
            "id": idx,
            "prompt": prompt,
            "ground_truth": answer,
            "rand-prompts": seeded_prompts,
            "correctness": correctness,
            "num_correct": num_correct,
            "pass_1024": pass_1024,
            "responses": all_responses
        })
        
        print(f"Problem {idx}: {num_correct}/{args.num_samples} correct, Pass@1024 = {pass_1024}")
    
    overall_pass_1024 = np.mean(all_pass_1024)
    total_correct = sum(item["num_correct"] for item in results)
    total_samples = len(results) * args.num_samples
    overall_accuracy = total_correct / total_samples
    
    print(f"\nOverall Results:")
    print(f"Pass@1024: {overall_pass_1024:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Total Problems: {len(results)}")
    print(f"Total Correct Answers: {total_correct}/{total_samples}")
    
    # Save results with metrics
    output_data = {
        "model_name": args.model_name,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "overall_metrics": {
            "pass_1024": overall_pass_1024,
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "num_problems": len(results)
        },
        "results": results
    }
    
    json_name = os.path.join(out_path, out_name + ".json")
    with open(json_name, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Also save a summary file
    summary_name = os.path.join(out_path, out_name + "_summary.txt")
    with open(summary_name, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Number of samples per problem: {args.num_samples}\n")
        f.write(f"Pass@1024: {overall_pass_1024:.4f}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Total Problems: {len(results)}\n")
        f.write(f"Total Correct Answers: {total_correct}/{total_samples}\n")
            
            
if __name__ == "__main__":
    args = parse_args()
    main(args)