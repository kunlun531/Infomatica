file_path=~/InForage/dataset/index
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=~/InForage/LLMs/e5-base-v2

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tools/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever
