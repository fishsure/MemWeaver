import argparse
import json
import os
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_phase", default='train')
parser.add_argument("--task", default='LaMP_1_time')
parser.add_argument("--ranker", default='recency')
parser.add_argument("--recency_topk", type=int, default=0)
parser.add_argument("--topk", type=int, default=30)

def print_args(args):
    for flag, value in args.__dict__.items():
        print(f'{flag}: {value}')

if __name__ == "__main__":
    opts = parser.parse_args()
    print_args(opts)

    task = opts.task
    ranker = opts.ranker
    result_file_name = 'rank_merge.json'

    # 输出路径构造
    if opts.recency_topk:
        output_ranking_addr = os.path.join(
            'data', task, f"{opts.data_phase}/{ranker}_{opts.topk}/{result_file_name}")
    else:
        output_ranking_addr = os.path.join(
            'data', task, f"{opts.data_phase}/{ranker}/{result_file_name}")

    # 加载 questions.json
    questions_path = os.path.join('data', task, f'{opts.data_phase}/{opts.data_phase}_questions.json')
    with open(questions_path, 'r') as file:
        dataset = json.load(file)

    print("\n[DEBUG] Loaded questions.json")
    print(f"Total questions: {len(dataset)}")
    print(f"First question sample:\n{json.dumps(dataset[0], indent=2, ensure_ascii=False)}")

    # 加载 outputs.json
    outputs_path = os.path.join('data', task, f'{opts.data_phase}/{opts.data_phase}_outputs.json')
    with open(outputs_path, 'r') as file:
        out_file = json.load(file)

    print("\n[DEBUG] Loaded outputs.json")
    if isinstance(out_file, dict) and 'golds' in out_file:
        out_dataset = out_file['golds']
        print("[INFO] Detected dict format with 'golds'")
        print(f"First gold output:\n{json.dumps(out_dataset[0], indent=2, ensure_ascii=False)}")
    elif isinstance(out_file, list):
        out_dataset = out_file
        print("[INFO] Detected list format")
        print(f"First output entry:\n{json.dumps(out_dataset[0], indent=2, ensure_ascii=False)}")
    else:
        raise ValueError("Unsupported format in outputs.json")

    # 校验 task 名称
    if isinstance(out_file, dict) and 'task' in out_file:
        assert task.startswith(out_file['task']), f"Task mismatch: {task} vs {out_file['task']}"

    # 排序与构造输出数据
    rank_dict = []
    for idx, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        assert data['id'] == out_dataset[idx]['id']
        profile = data['profile']
        user_id = data['user_id']

        # 加入 user_id 字段
        for entry in profile:
            entry['user_id'] = user_id

        # 时间排序
        profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
        ranked_profile = profile[::-1]  # 最新的排前面

        if opts.recency_topk:
            ranked_profile = ranked_profile[:opts.topk]

        rank_dict.append({
            'id': data['id'],
            'input': data['input'],
            'profile': ranked_profile,
            'user_id': user_id,
            'output': out_dataset[idx]['output']
        })

    # 保存结果
    print(f"\n[INFO] Saving ranked file to: {output_ranking_addr}")
    os.makedirs(os.path.dirname(output_ranking_addr), exist_ok=True)
    with open(output_ranking_addr, "w", encoding='utf-8') as file:
        json.dump(rank_dict, file, indent=4, ensure_ascii=False)
    print("[DONE]")
