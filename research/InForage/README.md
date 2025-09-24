<div align="center">
<h1>🔍 InForage: Scent of Knowledge</h1>
<h3>Optimizing Search-Enhanced Reasoning with Information Foraging</h3>

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red)](https://arxiv.org/abs/2505.09316)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS_2025-Spotlight-gold)](https://arxiv.org/abs/2505.09316)
[![Model](https://img.shields.io/badge/🤗_Model-InForage--3B--PPO-blue)](https://huggingface.co/TommyChien/InForage-3B-PPO)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-InForage_Data-green)](https://huggingface.co/datasets/TommyChien/InForage_data/)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)](LICENSE)

</div>

## 🔆 Overview

**InForage** is a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process, inspired by Information Foraging Theory (IFT). Unlike traditional static retrieval methods, InForage enables large language models to adaptively interact with external knowledge sources through iterative search behaviors.

### Key Features

- **🎯 Dynamic Information Foraging**: Adaptive inference-time retrieval that evolves with the reasoning process
- **🔄 Iterative Search & Reasoning**: Multi-step information gathering with quality-aware rewards
- **📊 Human-Guided Training Data**: Real-world web task trajectories capturing complex reasoning patterns
- **🏆 State-of-the-Art Performance**: Superior results on general QA, multi-hop reasoning, and real-time web QA

### Abstract

Augmenting large language models (LLMs) with external retrieval has become a standard method to address their inherent knowledge cutoff limitations. However, traditional retrieval-augmented generation methods employ static, pre-inference retrieval strategies, making them inadequate for complex tasks involving ambiguous, multi-step, or evolving information needs. Recent advances in test-time scaling techniques have demonstrated significant potential in enabling LLMs to dynamically interact with external tools, motivating the shift toward adaptive inference-time retrieval. 

Inspired by Information Foraging Theory (IFT), we propose **InForage**, a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process. Unlike existing approaches, InForage explicitly rewards intermediate retrieval quality, encouraging LLMs to iteratively gather and integrate information through adaptive search behaviors. To facilitate training, we construct a human-guided dataset capturing iterative search and reasoning trajectories for complex, real-world web tasks. 

Extensive evaluations across general question answering, multi-hop reasoning tasks, and a newly developed real-time web QA dataset demonstrate InForage's superior performance over baseline methods. These results highlight InForage's effectiveness in building robust, adaptive, and efficient reasoning agents.

## 📊 Dataset & Models

### 🤗 HuggingFace Resources

- **📦 [InForage Dataset](https://huggingface.co/datasets/TommyChien/InForage_data/)**: Complete training corpus and datasets
  - **Retrieval Corpus**: Training-time retrieval knowledge base
  - **SFT Data**: Cold-start supervised fine-tuning data
  - **Complex QA Data**: Human-guided complex question-answering trajectories

- **🎯 [InForage-3B-PPO Model](https://huggingface.co/TommyChien/InForage-3B-PPO)**: Our main experimental model trained with PPO



## 🔧 Training

### 📂 Process Training Data

For additional datasets such as **NQ** and **HotpotQA**, which were also used during training, you can obtain them from external resources, for example the [Search-R1 training data](https://github.com/PeterGriffinJin/Search-R1).  

After downloading, please place them under the `dataset` directory and follow the same preprocessing pipeline.  

```bash
python tasks/construct_rl_training_data.py
```

📝 SFT (Supervised Fine-Tuning)

We provide the cold-start SFT data in the dataset folder. You can run:

```bash
bash tasks/sft/sft.sh
```


🎮 RL (Reinforcement Learning with PPO)




### 🔍 Launch Retrieval Service

To start the retrieval service, use:

```bash
bash scripts/retrieval_launch.sh
```

- For Wikipedia retrieval, you can use the pre-built indexes from Search-R1 or FlashRAG.
- For the corpus constructed in this paper, you need to build the index in the same way, and then start another retrieval service with the same script.

In the configuration file, please provide:
- retriever.wiki_url: the URL endpoint of the Wikipedia retrieval service.
- retriever.url: the URL endpoint of the retrieval service for the corpus constructed in this paper.


Run PPO training with:

```bash
bash scripts/train_ppo.sh
```

This will launch reinforcement learning training with our default PPO configuration.


📈 Evaluate

We provide evaluation scripts for all benchmark datasets:

```bash
bash scripts/eval.sh
```
Evaluation results will be logged under outputs/.

✍️ Annotation System

We also provide an annotation system for creating new human-guided trajectories. You can launch the annotation interface with:

```bash
streamlit run annotation/annotate_page.py
```

This tool supports sentence-level annotation and trajectory construction for iterative search tasks.


### 📚 Citation

If you find InForage useful in your research, please cite our paper:
```bibtex
@misc{qian2025scentknowledgeoptimizingsearchenhanced,
      title        = {Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging}, 
      author       = {Hongjin Qian and Zheng Liu},
      year         = {2025},
      eprint       = {2505.09316},
      archivePrefix= {arXiv},
      primaryClass = {cs.CL},
      url          = {https://arxiv.org/abs/2505.09316}
}
```


