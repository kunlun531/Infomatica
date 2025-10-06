import json
import time
import traceback
from multiprocessing import Pool
from openai import OpenAI
import os

# NOTE: Input research tree JSONL file.
INPUT_FILE = "/share/project/kunluo/Datasets/Reasoning/ConstructedTreeData/0724--DeepSeek--hopnum5to8--Num18K.jsonl"
# Output path
PASS_FILE = "/share/project/kunluo/Datasets/Reasoning/ConstructedTreeData/DiffFilterPass--0724--DeepSeek--hopnum5to8--Num18K.jsonl"
FAIL_FILE = "/share/project/kunluo/Datasets/Reasoning/ConstructedTreeData/DiffFilterFail--0724--DeepSeek--hopnum5to8--Num18K.jsonl"

# --- API 初始化函数 (每个子进程独立调用) ---
def init_client():
    """初始化并返回一个OpenAI客户端实例。"""
    return OpenAI(
        api_key="sk-fd5d5a4458064f76a5cc34e9b386c17e", # 请确保这是一个有效的API Key
        base_url="https://api.deepseek.com"
    )

# --- 评估Prompt模板 (内容保持不变) ---
PROMPT_TEMPLATE = """
**Your Role:** You are a pragmatic AI Dataset Quality Analyst. Your task is to evaluate complex, multi-hop search-based questions to determine if they are well-formed and genuinely require a multi-step search process to solve. **The dataset aims to train LLM to conduct Deep Research.**

**Primary Objective:** Your goal is to filter for high-quality questions that cannot be easily answered by an LLM's internal knowledge and instead encourage a step-by-step investigation using a search tool. Aim for a balanced evaluation (not too strict), targeting a pass rate of approximately 40%.

### **Key Evaluation Criteria**
Evaluate each question based on the following principles. A strong question should generally meet these standards.

1.  **Meaningful Multi-Step Process:**
    *   The question should require **more than two distinct reasoning or search steps** to solve.
    *   Most of these steps must necessitate an **external search**, not just rely on common knowledge.

2.  **Logical Chain:**
    *   The reasoning process must be cohesive. Each step should logically depend on the information found in the previous one. Avoid questions that are just a list of disconnected facts.

3.  **Clarity and Uniqueness:**
    *   The question must be phrased clearly, without ambiguity or grammatical errors.
    *   It must point to a **single, verifiable answer**. Questions with subjective or multiple correct answers should be rejected.

4.  **Resistance to Shortcuts (Leakage Test):**
    *   This is a critical check. The question **fail** if a single, well-crafted search query can bypass the intended multi-step path and lead directly to the answer. Pay attention to "information leakage."

### **Your Output Format**
You should return your evaluation using the following **JSON-style structure strictly**:
```json
{{
  "Question": "{question}",
  "Analysis": "[Concise justification referencing the guidelines above.]",
  "Decision": "[Pass or Fail]"
}}
```
"""

# --- 多线程工作函数：评估单个数据项 ---
def evaluate_data_item(index_data_tuple):
    """
    对单个数据项进行API评估，具备全面的错误处理机制。
    任何失败都会导致返回'Fail'决策。
    """
    index, data_item = index_data_tuple
    
    # 最终的安全网，捕获任何意料之外的错误
    try:
        # 安全地提取问题
        question = data_item.get('root', {}).get('question')
        if not question:
            print(f"[{index+1}] ❌ FAIL: Question is missing or empty in data item.")
            return {'original_data': data_item, 'decision': 'Fail', 'error': 'MissingQuestion'}

        client = init_client()
        user_content = f"Please evaluate the following question based on the provided criteria:\n\n---\n{question}\n---\n\n{PROMPT_TEMPLATE}"
        
        print(f"[{index+1}] 🟡 API-Eval Start: \"{question[:60].strip()}...\"")
        start_time = time.time()

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": user_content}],
            stream=False,
            temperature=0.3,  # 使用较低温度以获得更稳定的JSON输出
        )

        result_text = response.choices[0].message.content.strip()
        elapsed = time.time() - start_time
        
        # 尝试解析API返回的JSON结果
        try:
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            
            evaluation_json = json.loads(result_text)
            # 规范化Decision字段，确保只有"Pass"才算通过
            decision = evaluation_json.get("Decision", "Fail").strip().capitalize()
            if decision != "Pass":
                decision = "Fail"

            print(f"[{index+1}] ✅ API-Eval Done in {elapsed:.2f}s. Decision: {decision}")
            return {'original_data': data_item, 'decision': decision, 'error': None}
            
        except (json.JSONDecodeError, AttributeError):
            print(f"[{index+1}] ❌ FAIL: API response is not valid JSON. Elapsed: {elapsed:.2f}s.")
            print(f"[{index+1}] 📝 Raw Output:\n{result_text}\n{'-'*60}")
            return {'original_data': data_item, 'decision': 'Fail', 'error': 'InvalidJSONResponse'}

    except Exception as e:
        print(f"[{index+1}] ❌ FAIL: An unexpected error occurred during API evaluation.")
        traceback.print_exc()
        return {'original_data': data_item, 'decision': 'Fail', 'error': str(e)}

# --- 主函数 ---
def main():
    main_start_time = time.time()
    
    items_to_evaluate_api = []
    hop_fail_list = []
    parsing_fail_count = 0

    print(f"📖 Step 1: Reading and pre-filtering data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data_item = json.loads(line)
                    # 安全地获取hop_num，如果不存在则默认为0
                    hop_num = data_item.get('hop_num', 0)
                    
                    # 预筛选：判断hop_num
                    if hop_num <= 2:
                        hop_fail_list.append(data_item)
                    else:
                        items_to_evaluate_api.append(data_item)
                except json.JSONDecodeError:
                    print(f"⚠️ WARNING: Skipping malformed JSON on line {i+1}.")
                    parsing_fail_count += 1
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Input file not found at {INPUT_FILE}")
        return

    print("📊 Pre-filtering complete.")
    print(f"  - {len(hop_fail_list)} items REJECTED (hop_num <= 2).")
    print(f"  - {parsing_fail_count} lines SKIPPED due to JSON parsing errors.")
    print(f"  - {len(items_to_evaluate_api)} items will be sent for API evaluation.")

    # 将因hop_num不足而失败的数据首先写入Fail文件 (覆盖模式)
    print(f"💾 Writing {len(hop_fail_list)} hop-filtered items to {FAIL_FILE}...")
    with open(FAIL_FILE, 'w', encoding='utf-8') as f_fail:
        for item in hop_fail_list:
            f_fail.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    if not items_to_evaluate_api:
        print("\n✅ No items remaining for API evaluation. Process finished.")
        return

    # 步骤2: 使用多线程调用API进行评估
    indexed_items = list(enumerate(items_to_evaluate_api))
    process_num = 200  # 使用的进程数
    print(f"\n🚀 Step 2: Starting parallel API evaluation for {len(indexed_items)} items using {process_num} processes...")

    with Pool(processes=process_num) as pool:
        results = pool.map(evaluate_data_item, indexed_items)

    print("\n✅ API evaluation complete.")
    print(f"💾 Step 3: Sorting and saving API results...")

    # 步骤3: 根据API评估结果，将数据分别写入Pass和Fail文件
    api_pass_count = 0
    api_fail_count = 0
    with open(PASS_FILE, 'w', encoding='utf-8') as f_pass, \
         open(FAIL_FILE, 'a', encoding='utf-8') as f_fail:  # 追加模式写入Fail文件
        
        for result in results:
            if result and result.get('decision') == 'Pass':
                f_pass.write(json.dumps(result['original_data'], ensure_ascii=False) + '\n')
                api_pass_count += 1
            else:
                f_fail.write(json.dumps(result['original_data'], ensure_ascii=False) + '\n')
                api_fail_count += 1

    # --- 最终总结 ---
    total_fail_count = len(hop_fail_list) + api_fail_count
    total_processed = api_pass_count + total_fail_count
    
    print("\n--- FINAL SUMMARY ---")
    print(f"  - Total items processed: {total_processed}")
    print(f"  - Passed (API eval): {api_pass_count}")
    print(f"  - Failed (Total): {total_fail_count}")
    print(f"    - Failed by hop_num <= 3: {len(hop_fail_list)}")
    print(f"    - Failed by API eval / Error: {api_fail_count}")
    print(f"  - Skipped (Malformed JSON): {parsing_fail_count}")
    print("  --------------------")
    print(f"  -> Results PASSED saved to: {PASS_FILE}")
    print(f"  -> Results FAILED saved to: {FAIL_FILE}")

    main_end_time = time.time()
    total_elapsed = main_end_time - main_start_time
    minutes, seconds = divmod(total_elapsed, 60)
    print(f"\n🎉 Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
