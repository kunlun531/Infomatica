# FilterSFTData.py
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# OpenAI 客户端初始化
client = OpenAI(
    api_key="sk-fd5d5a4458064f76a5cc34e9b386c17e",
    base_url="https://api.deepseek.com"
)

# 输入输出路径
input_file = "/share/project/kunluo/Datasets/Reasoning/SFTData/RFT--FromRLModel--Source-WikiHard32K-NQHotpot6K--0815.jsonl"
output_file = "/share/project/kunluo/Datasets/Reasoning/SFTData/RFT--FromRLModel--Source-WikiHard32K-NQHotpot6K--0815--QualityAssure.jsonl"

# 过滤提示词
input_prompt_template = """
You are tasked with evaluating whether a trajectory is high-quality training data for Supervised Fine-Tuning (SFT) of an LLM as a Deep Research Agent. Assess it under the following strict criteria:

1. **Step-by-Step Reasoning**
- The agent must break the question into sub-problems and reason progressively toward the answer.
- Direct shortcuts to the answer without multi-hop reasoning are not allowed.

2. **Search Quality**
- Searches must be justified by prior reasoning, not speculative guesswork.
- Trajectories where the question is trivially answerable by a single query (too easy, no real reasoning required) must be rejected.

3. **Leakage and Shortcut Avoidance**
- The question must not contain conditions that uniquely identify the answer (e.g., “first licensed nurse in Colorado”), turning the task into a lookup.
- “Additionally” clauses should not leak extra identifiers that make the answer trivially searchable.
- High-quality trajectories should require genuine multi-step exploration and synthesis.

5. **Output Format**
- First, provide a concise evaluation of whether the trajectory meets the above criteria.
- Then, on a new line, output only the verdict in the form: \\boxed{{Pass}} or \\boxed{{Failed}}.

Trajectory:
{text}
"""

def evaluate_trajectory(conversation):
    """Call API to evaluate a single data item"""
    try:
        # Concatenate user + assistant to form full text
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            text += f"{role.upper()}:\n{content}\n\n"

        # Randomly decide whether to log (2% chance)
        if random.random() < 0.005:
            print("\n" + "="*60)
            print("LOGGING SAMPLE (2% probability)")
            print("Input:")
            print(text[:300] + "..." if len(text) > 300 else text)
            print("-" * 60)

        # Call API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": input_prompt_template.format(text=text)}],
            stream=False,
            temperature=0.5,
        )

        output = response.choices[0].message.content.strip()

        # Log if this sample is selected (2%)
        if random.random() < 0.01:
            print("Full API Output:")
            print(output)
            print("="*60)

        if "\\boxed{Pass}" in output:
            return True  # Keep
        else:
            return False  # Filter out
    except Exception as e:
        print(f"Error processing trajectory: {e}")
        return False


def main():
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # data = data[:100]  # Limit for testing
    results = []

    # Multi-threaded processing
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_item = {executor.submit(evaluate_trajectory, item["conversation"]): item for item in data}

        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Filtering"):
            item = future_to_item[future]
            try:
                keep = future.result()
                if keep:
                    results.append(item)
            except Exception as e:
                print(f"Future exception: {e}")

    # Write output file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Finished. Preserved {len(results)} / {len(data)} high quality data.")


if __name__ == "__main__":
    main()
