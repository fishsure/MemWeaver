python ranking.py \
    --task LaMP_1_time \
    --data_split dev \
    --rank_stage mem_graph \
    --base_retriever_path models/bge-m3 \
    --llm_name models/Llama-3.1-8B-Instruct \
    --CUDA_VISIBLE_DEVICES 1 \
    --end_idx 1500 \
    --topk 5 \
    --use_kmeans 0 \
    --use_recency 0 \
    --time_decay_lambda 0 \
    --use_graph_walk 1 \
    --walk_start_method semantic \
    --walk_length 15 \
    --semantic_alpha 1.5 \
    --time_lambda1 0.01 \
    --time_lambda2 0.01 

python generation/generate.py \
    --CUDA_VISIBLE_DEVICES 1 \
    --task LaMP_1_time \
    --input_path dev/recency/bge-m3_5/ \
    --source retrieval \
    --file_name file_name \
    --model_name models/Llama-3.1-8B-Instruct