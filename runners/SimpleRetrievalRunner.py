import copy
import json
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn
class SimpleRetriever:

    @staticmethod
    def parse_args(parser):
        # BGE retriever related arguments
        parser.add_argument("--base_retriever_path",
                            default="/data/yu12345/models/bge-m3")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)

        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')

        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split
        self.source = opts.source
        self.topk = opts.topk
        self.device = opts.device
        self.batch_size = opts.batch_size

        # Load user data
        self.load_user(opts)
        
        # Initialize BGE retriever
        self.retriever = RetrieverModel(
            ret_type='dense',
            model_path=opts.base_retriever_path,
            base_model_path=opts.base_retriever_path,
            user2id=self.user2id,
            user_emb_path=self.user_emb_path,
            batch_size=self.batch_size,
            device=self.device,
            max_length=opts.max_length,
            pooling=opts.retriever_pooling,
            normalize=opts.retriever_normalize).eval().to(self.device)

        # Load dataset
        input_path = os.path.join(self.data_addr, opts.data_split, self.source,
                                  'rank_merge.json')

        self.dataset = json.load(open(input_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def load_user(self, opts):
        """Load user vocabulary and embeddings"""
        vocab_addr = os.path.join(opts.data_addr, f"dev/{opts.source}")

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert len(self.user_vocab) == len(self.user2id)

        self.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          "20241009-120906.pt")

    def run(self):
        """Run the memorizer to retrieve relevant profiles"""
        # Determine output directory and file name
        retriever_name = "bge-m3"
        sub_dir = f"{retriever_name}_{self.topk}"
        file_name = f"simple_base"

        results = []
        for data in tqdm(self.dataset):
            query, selected_profs = self.retrieve_topk(data['input'],
                                                       data['user_id'])
            result_item = {
                "input": data['input'],
                "query": query,
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": selected_profs
            }
            results.append(result_item)

        output_addr = os.path.join(self.output_addr, self.data_split,
                                   self.source, sub_dir, 'retrieval')

        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{file_name}.json")
        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    def retrieve_topk(self, inp, user):
        """Retrieve top-k profiles for given input and user"""
        # Get current user's profile
        user_id = self.user2id[user]
        current_profile = self.user_vocab[user_id]['profile']
        
        query = self.get_query(inp)
        
        # Check if query is None and handle it
        if query is None:
            print(f"Warning: get_query returned None for input: {inp}")
            query = str(inp)  # Use input as fallback query
        
        cur_corpus = self.get_corpus(current_profile, self.use_date)
        cur_retrieved, cur_scores = self.retrieve_topk_one_user(
            cur_corpus, current_profile, query, user, self.topk)
        
        all_retrieved = []
        for data_idx, data in enumerate(cur_retrieved):
            cur_data = copy.deepcopy(data)
            if self.task.startswith('LaMP_3'):
                cur_data['rate'] = cur_data['score']
            cur_data['score'] = cur_scores[data_idx]
            all_retrieved.append(cur_data)
            
        return query, all_retrieved

    def retrieve_topk_one_user(self, corpus, profile, query, user, topk):
        """Retrieve top-k items for one user using BGE dense retrieval"""
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
            print(f"Warning: Query converted to string: {query}")
        
        # Ensure corpus is a list of strings
        if not isinstance(corpus, list):
            print(f"Warning: Corpus is not a list: {type(corpus)}")
            corpus = [str(corpus)] if corpus is not None else [""]
        
        # Ensure profile is a list
        if not isinstance(profile, list):
            print(f"Warning: Profile is not a list: {type(profile)}")
            profile = [profile] if profile is not None else [{}]
        
        # Use BGE retrieval directly
        selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
            corpus, profile, query, user, topk
        )
        
        return selected_profs, dense_scores

 