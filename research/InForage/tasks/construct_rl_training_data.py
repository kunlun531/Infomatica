from my_own_tools import *
import random
from datasets import Dataset, Features, Value
import os
def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""You are an intelligent agent designed to solve complex queries or tasks by retrieving external knowledge and reasoning step by step.

Please follow these instructions carefully:

1. For each piece of information you receive:
   - Think step by step and explain your reasoning inside <think> and </think> tags.

2. If you need more information to proceed:
   - Issue a search by writing your subquery inside <search> and </search> tags.
   - Retrieved results will appear between <evidence> and </evidence> tags.
   - You can conduct multiple searches as needed.

3. When you have collected enough information:
   - Provide your final answer using the <answer> and </answer> tags.
   - Do not include explanations or reasoning in the answer block.
   - Keep your answer concise.

Now, solve the following task:

Task: {question}
"""
    else:
        raise NotImplementedError
    return prefix

if __name__ == '__main__':
    data_path = "~/InForage/dataset/generated_complex_qa"
    complex_query = load_jsonl(os.path.join(data_path, 'complex_queries.jsonl'))
    multi_query = load_jsonl(os.path.join(data_path, 'multihop_queries.jsonl'))
    claim_set = load_jsonl(os.path.join(data_path, 'claim_sets.jsonl'))
    id2claim = {claim['claim_set_id']: [c["url"] for c in claim['claim']] for claim in claim_set}

    all_query = []

    for q in complex_query+multi_query:
        query = q['query']['query']
        answer = q['query']['answer']
        claim_set_id = q['claim_set_id']
        if claim_set_id not in id2claim:
            continue
        if not answer or not query:
            continue
        query_type = q['type']
        all_query.append({
            'question': query,
            'answer': answer,
            'query_type': query_type,
            'source_urls': id2claim[claim_set_id],
        })

    train_query = all_query[500:]
    test_query = all_query[:500]
    data_source = 'self'
    train_dataset = Dataset.from_list(train_query)
    test_dataset = Dataset.from_list(test_query)
    local_dir = "~/InForage/dataset/train"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            question = make_prefix(example, template_type='base')
            solution = {
                "target": example['answer'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'type': example['query_type'],
                    'index': idx,
                    'source_urls': example['source_urls'],
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    

    

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
