import copy
import json
import os
import pickle
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn
from runners.agent_llm_utils import agent_llm_summarize_summary, agent_llm_summarize_json, agent_llm_summarize_tool_call_end2end_json


class MemRetriever:

    @staticmethod
    def parse_args(parser):
        parser.add_argument("--ret_type",
                            default="dense_tune",
                            choices=[
                                'zero_shot', 'random', 'recency', 'bm25',
                                'dense', 'dense_tune'
                            ])

        parser.add_argument("--base_retriever_path",
                            default="/data/yu12345/models/bge-base-en-v1.5")
        parser.add_argument("--retriever_checkpoint",
                            default="bge-base-en-v1.5/20250606-194643")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)

        parser.add_argument("--retrieve_user", type=int, default=1)

        parser.add_argument("--user_emb_path", default="20250529-092233.pt")
        parser.add_argument("--user_vocab_path", default="")

        parser.add_argument("--user_topk", type=int, default=6)
        parser.add_argument("--n_clusters", type=int, default=10)
        # 新增参数
        parser.add_argument("--use_kmeans", type=int, default=1)
        parser.add_argument("--kmeans_select_method", default="center", choices=["center", "relevance"], 
                          help="K-means cluster representative selection method: center (closest to centroid) or relevance (most relevant to query)")
        parser.add_argument("--use_recency", type=int, default=1)
        # 新增参数
        parser.add_argument("--time_decay_lambda", type=float, default=0.0)
        # 新增参数 for agent summarization
        parser.add_argument("--agent", type=int, default=0, help="Whether to use LLM to summarize user preferences")
        parser.add_argument("--agent_k", type=int, default=5, help="Number of profile records per LLM summarization step")
        parser.add_argument("--agent_model_name", default="Qwen2-7B-Instruct", help="LLM model name for agent summarization")
        parser.add_argument("--agent_mode", default="summary", choices=["summary", "json", "tool_call"], help="LLM agent mode: 'summary' for direct summary, 'json' for structured JSON, 'tool_call' for tool-calling style (end-to-end JSON)")
        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')
        self.llm_name = opts.llm_name

        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split
        self.source = opts.source
        self.ret_type = opts.ret_type
        self.topk = opts.topk
        self.retrieve_user = opts.retrieve_user
        self.device = opts.device
        self.n_clusters = opts.n_clusters
        # 新增参数
        self.use_kmeans = getattr(opts, "use_kmeans", 1)
        self.kmeans_select_method = getattr(opts, "kmeans_select_method", "center")
        self.use_recency = getattr(opts, "use_recency", 1)
        self.time_decay_lambda = getattr(opts, "time_decay_lambda", 0.0)
        # agent summarization params
        self.agent = getattr(opts, "agent", 0)
        self.agent_k = getattr(opts, "agent_k", 5)
        self.agent_model_name = getattr(opts, "agent_model_name", "Qwen2-7B-Instruct")
        self.agent_mode = getattr(opts, "agent_mode", "summary")
        if self.agent:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            self.agent_model_path = f'/data/yu12345/models/{self.agent_model_name}'
            self.agent_tokenizer = AutoTokenizer.from_pretrained(self.agent_model_path)
            self.agent_tokenizer.padding_side = "left"
            if self.agent_tokenizer.pad_token_id is None:
                self.agent_tokenizer.pad_token = self.agent_tokenizer.eos_token
                self.agent_tokenizer.pad_token_id = self.agent_tokenizer.eos_token_id
            self.agent_llm = LLM(model=self.agent_model_path, gpu_memory_utilization=0.9, max_model_len=8192)
            self.agent_sampling_params = SamplingParams(seed=42, temperature=0, best_of=1, max_tokens=500)
        self.load_user(opts)

        if self.ret_type == 'dense' or self.ret_type == 'dense_tune':
            self.batch_size = opts.batch_size

            if self.ret_type == 'dense':
                self.retriever_checkpoint = opts.base_retriever_path
            elif self.ret_type == 'dense_tune':
                opts.retriever_checkpoint = os.path.join(
                    opts.output_addr, f"train/{opts.source}",
                    opts.retriever_checkpoint)
                self.retriever_checkpoint = opts.retriever_checkpoint

            self.retriever = RetrieverModel(
                ret_type=self.ret_type,
                model_path=self.retriever_checkpoint,
                base_model_path=opts.base_retriever_path,
                user2id=self.user2id,
                user_emb_path=opts.user_emb_path,
                batch_size=self.batch_size,
                device=self.device,
                max_length=opts.max_length,
                pooling=opts.retriever_pooling,
                normalize=opts.retriever_normalize).eval().to(self.device)

        input_path = os.path.join(self.data_addr, opts.data_split, self.source,
                                  'rank_merge.json')

        self.dataset = json.load(open(input_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

        self.bm25_debug_log = []

    def load_user(self, opts):
        opts.user_vocab_path = os.path.join(opts.data_addr,
                                            f"dev/{opts.source}")
        vocab_addr = opts.user_vocab_path

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert len(self.user_vocab) == len(self.user2id)

        opts.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          opts.user_emb_path)

        self.user_emb_path = opts.user_emb_path

        if self.retrieve_user:
            self.user_emb_name = '.'.join(
                os.path.basename(self.user_emb_path).split('.')[:-1])
            self.user_embedding = torch.load(self.user_emb_path).to(
                self.device)
            self.user_topk = opts.user_topk

            assert self.user_embedding.shape[0] == len(self.user_vocab)

    def summarize_user_profile(self, profile):
        """
        Summarize user profile using the selected agent_mode.
        Supports: 'summary' (natural language), 'json' (structured JSON), 'tool_call' (tool-call style, end-to-end JSON)
        """
        k = self.agent_k
        # For summary mode, use string; for json/tool_call, use list (or dict if needed)
        if self.agent_mode == "summary":
            summary = ""
        else:
            summary = []
        for i in range(0, len(profile), k):
            records = profile[i:i+k]
            record_texts = [str(r) for r in records]
            # 新增：打印输入长度和token数
            input_text = "\n".join(record_texts)
            tokens = self.agent_tokenizer(input_text, return_tensors="pt")
            num_tokens = tokens["input_ids"].shape[1]
            # print(f"[DEBUG] agent_k={k}, 输入文本长度={len(input_text)}，token数={num_tokens}，tokenizer最大长度={self.agent_tokenizer.model_max_length}")
            if self.agent_mode == "summary":
                summary = agent_llm_summarize_summary(summary, record_texts, self.agent_tokenizer, self.agent_llm, self.agent_sampling_params)
            elif self.agent_mode == "json":
                # Use agent_llm_summarize_json, output is JSON string
                summary = agent_llm_summarize_json(summary, record_texts, self.agent_tokenizer, self.agent_llm, self.agent_sampling_params, task_type=self.task)
            elif self.agent_mode == "tool_call":
                # Use tool_call end-to-end, output is JSON string
                summary = agent_llm_summarize_tool_call_end2end_json(summary, record_texts, self.agent_tokenizer, self.agent_llm, self.agent_sampling_params, task_type=self.task)
            else:
                raise ValueError(f"Unknown agent_mode: {self.agent_mode}")
        print(f"[MemRetriever] Used agent_mode: {self.agent_mode}")
        return summary

    def run(self):
        user_summaries = {}
        if self.agent:
            for data in self.dataset:
                user_id = data['user_id']
                user_idx = self.user2id[user_id]
                profile = self.user_vocab[user_idx]['profile']
                # 只为每个 user_id 生成一次 summary
                if user_id not in user_summaries:
                    summary = self.summarize_user_profile(profile)
                    user_summaries[user_id] = summary
        if self.ret_type == 'zero_shot':
            sub_dir = self.ret_type
            file_name = "mem_base"
        else:
            if self.ret_type in ['random', 'recency', 'bm25']:
                sub_dir = f"{self.ret_type}_{self.topk}"
                file_name = "mem_base"
            elif self.ret_type == 'dense':
                sub_dir = f"{self.retriever_checkpoint.split('/')[-1]}_{self.topk}"
                file_name = "mem_base"
            elif self.ret_type == 'dense_tune':
                retriever_name = self.retriever_checkpoint.split('/')[-2]
                train_time = self.retriever_checkpoint.split('/')[-1]
                sub_dir = f"{retriever_name}_{self.topk}"
                file_name = f"{train_time}"

            if self.retrieve_user:
                file_name += '_user-{}_{}'.format(self.user_topk,
                                                  self.user_emb_name)
            file_name += f'_kmeans-{self.use_kmeans}_method-{self.kmeans_select_method}_clusters-{self.n_clusters}_recency-{self.use_recency}_decay-{self.time_decay_lambda}_agent-{self.agent}_agentk-{self.agent_k}_agent_mode-{self.agent_mode}'

        results = []
        for data in tqdm(self.dataset):
            query, selected_profs = self.retrieve_topk(data['input'],
                                                       data['user_id'])
            # 获取summary（如果有agent）
            summary = user_summaries[data['user_id']] if self.agent else None
            results.append({
                "input": data['input'],
                "query": query,
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": selected_profs,
                "summary": summary
            })

        output_addr = os.path.join(self.output_addr, self.data_split,
                                   self.source, sub_dir, 'retrieval')

        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{file_name}.json")
        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        # save debug log for BM25
        if self.ret_type == "bm25" and len(self.bm25_debug_log) > 0:
            debug_path = 'bm25_rank_debug.json'
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(self.bm25_debug_log, f, indent=2, ensure_ascii=False)

    def retrieve_topk(self, inp, user):
        all_profiles = self.retrieve_user_topk(user)

        query = self.get_query(inp)
        all_retrieved = []
        for i in range(len(all_profiles)):
            cur_corpus = self.get_corpus(all_profiles[i], self.use_date)
            cur_retrieved, cur_scores = self.retrieve_topk_one_user(
                cur_corpus, query, all_profiles[i], user, self.topk)
            new_cur_retrieved = []
            for data_idx, data in enumerate(cur_retrieved):
                cur_data = copy.deepcopy(data)
                if self.task.startswith('LaMP_3'):
                    cur_data['rate'] = cur_data['score']
                cur_data['score'] = cur_scores[data_idx]
                new_cur_retrieved.append(cur_data)
            all_retrieved.extend(new_cur_retrieved)
        return query, all_retrieved

    def _get_time_weight(self, profile_list, ref_date=None):
        """
        profile_list: list of dict, each with 'date' field (YYYY-MM-DD)
        ref_date: datetime.date, reference date. If None, use max(profile_list)
        Returns: list of float, time weights
        """
        if ref_date is None:
            # 以所有profile中最新的时间为参考
            ref_date = max(
                [datetime.datetime.strptime(p['date'], "%Y-%m-%d").date() for p in profile_list]
            )
        weights = []
        for p in profile_list:
            d = datetime.datetime.strptime(p['date'], "%Y-%m-%d").date()
            delta_days = (ref_date - d).days
            weight = np.exp(-self.time_decay_lambda * delta_days)
            weights.append(weight)
        return weights

    def retrieve_topk_one_user(self, corpus, query, profile, user, topk):
        n_clusters = self.n_clusters

        # 只要不是 random/recency/zero_shot，才用下面的逻辑
        if self.ret_type in ["bm25", "dense", "dense_tune"]:
            # 1. kmeans
            if self.use_kmeans:
                if self.ret_type == "bm25":
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(corpus).toarray()
                else:
                    with torch.no_grad():
                        corpus_tokens = self.retriever.tokenizer(
                            corpus,
                            padding=True,
                            truncation=True,
                            max_length=self.retriever.max_length,
                            return_tensors='pt'
                        ).to(self.device)
                        corpus_emb = self.retriever.encode(corpus_tokens)
                    X = corpus_emb.cpu().numpy()
                kmeans = KMeans(n_clusters=min(n_clusters, len(corpus)), random_state=42, n_init=10).fit(X)
                cluster_labels = kmeans.labels_
                
                # 从每个聚类中选择代表样本
                representative_indices = []
                for i in range(kmeans.n_clusters):
                    cluster_idx = np.where(cluster_labels == i)[0]
                    if len(cluster_idx) == 0:
                        continue
                    
                    if self.kmeans_select_method == "center":
                        # 选择最接近聚类中心的样本
                        if self.ret_type == "bm25":
                            cluster_vectors = X[cluster_idx]
                        else:
                            cluster_vectors = X[cluster_idx]
                        center_vector = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                        selected_idx = cluster_idx[np.argmin(distances)]
                    elif self.kmeans_select_method == "relevance":
                        # 选择与查询最相关的样本作为聚类代表
                        cluster_corpus = [corpus[idx] for idx in cluster_idx]
                        cluster_profile = [profile[idx] for idx in cluster_idx]
                        
                        try:
                            if self.ret_type == "bm25":
                                # 对于BM25，使用TF-IDF计算相关性
                                tokenized_cluster = [x.split() for x in cluster_corpus]
                                bm25_cluster = BM25Okapi(tokenized_cluster)
                                cluster_scores = bm25_cluster.get_scores(query.split())
                                best_idx = np.argmax(cluster_scores)
                                selected_idx = cluster_idx[best_idx]
                            else:
                                # 对于dense检索，使用检索器计算相关性
                                cluster_profs, cluster_scores = self.retriever.retrieve_topk_dense(
                                    cluster_corpus, cluster_profile, query, user, len(cluster_corpus)
                                )
                                best_idx = np.argmax(cluster_scores)
                                selected_idx = cluster_idx[best_idx]
                        except Exception as e:
                            print(f"[K-means Relevance Selection] 聚类 {i} 相关性选择失败: {e}")
                            # 回退到中心选择
                            if self.ret_type == "bm25":
                                cluster_vectors = X[cluster_idx]
                            else:
                                cluster_vectors = X[cluster_idx]
                            center_vector = kmeans.cluster_centers_[i]
                            distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                            selected_idx = cluster_idx[np.argmin(distances)]
                    else:
                        # 默认使用中心选择
                        if self.ret_type == "bm25":
                            cluster_vectors = X[cluster_idx]
                        else:
                            cluster_vectors = X[cluster_idx]
                        center_vector = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                        selected_idx = cluster_idx[np.argmin(distances)]
                    
                    representative_indices.append(selected_idx)
                
                kmeans_corpus = [corpus[i] for i in representative_indices]
                kmeans_profile = [profile[i] for i in representative_indices]
            else:
                kmeans_corpus = []
                kmeans_profile = []

            # 2. recency
            if self.use_recency:
                recency_sorted = sorted(
                    zip(corpus, profile),
                    key=lambda x: tuple(map(int, str(x[1]['date']).split("-"))),
                    reverse=True
                )
                recency_corpus = [x[0] for x in recency_sorted[:topk]]
                recency_profile = [x[1] for x in recency_sorted[:topk]]
            else:
                recency_corpus = []
                recency_profile = []

            # 3. 合并
            if self.use_kmeans or self.use_recency:
                merged_corpus = kmeans_corpus + [c for c in recency_corpus if c not in kmeans_corpus]
                merged_profile = kmeans_profile + [p for p in recency_profile if p not in kmeans_profile]
            else:
                merged_corpus = corpus
                merged_profile = profile

            # 4. 检索
            if self.ret_type == "bm25":
                tokenized_corpus = [x.split() for x in merged_corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                scores = bm25.get_scores(query.split())
                # 时间权重
                if self.time_decay_lambda > 0:
                    time_weights = self._get_time_weight(merged_profile)
                    scores = np.array(scores) * np.array(time_weights)
                sorted_idx = np.argsort(scores)[::-1][:topk]
                top_n_scores = [scores[i] for i in sorted_idx]
                selected_profs = [merged_profile[i] for i in sorted_idx]

                # save debug info
                if len(self.bm25_debug_log) < 10:
                    before = [{"text": merged_profile[i], "score": float(scores[i])} for i in range(min(5, len(merged_profile)))]
                    after = [{"text": merged_profile[i], "score": float(scores[i])} for i in sorted_idx[:5]]
                    self.bm25_debug_log.append({
                        "query": query,
                        "before": before,
                        "after": after
                    })
            else:  # dense/dense_tune
                dense_scores = None
                selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
                    merged_corpus, merged_profile, query, user, len(merged_corpus)
                )
                # dense_scores: 检索器返回的分数，顺序与selected_profs一致
                # 需要加时间权重再排序
                if self.time_decay_lambda > 0:
                    time_weights = self._get_time_weight(selected_profs)
                    dense_scores = np.array(dense_scores) * np.array(time_weights)
                # 重新排序
                sorted_idx = np.argsort(dense_scores)[::-1][:topk]
                selected_profs = [selected_profs[i] for i in sorted_idx]
                top_n_scores = [dense_scores[i] for i in sorted_idx]

        elif self.ret_type == "random":
            selected_profs = random.choices(profile, k=topk)
            top_n_scores = [1.0] * topk
        elif self.ret_type == "recency":
            profile = sorted(
                profile,
                key=lambda x: tuple(map(int, str(x['date']).split("-"))))
            randked_profile = profile[::-1]
            selected_profs = randked_profile[:topk]
            top_n_scores = [1.0] * topk
        elif self.ret_type == 'zero_shot':
            selected_profs = []
            top_n_scores = []

        return selected_profs, top_n_scores

    def retrieve_user_topk(self, user):
        user_id = self.user2id[user]
        if self.retrieve_user:
            cur_user_emb = self.user_embedding[[user_id]]
            sims = F.cosine_similarity(cur_user_emb, self.user_embedding)
            topk_values, topk_indices = torch.topk(sims, self.user_topk * 2)

            top_k_user_id = topk_indices.tolist()[:self.user_topk]
            topk_scores = [sims[i].item() for i in top_k_user_id]
        else:
            topk_scores = [1]
            top_k_user_id = [user_id]

        topk_profile = []
        for idx, user_idx in enumerate(top_k_user_id):
            cur_profile = self.user_vocab[user_idx]['profile']
            new_profile = []
            for data in cur_profile:
                new_data = copy.deepcopy(data)
                new_data['user_sim'] = topk_scores[idx]
                new_profile.append(new_data)
            topk_profile.append(new_profile)

        return topk_profile
