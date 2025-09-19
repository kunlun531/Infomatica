# InForage


## Process training data

We only provide the self-constructed training dataset. For NQ and HotpotQA that also used for training, we do not provide the data here due to size limit. You can download the data from the other source such as using FlashRAG.

```bash
python tasks/construct_rl_training_data.py
```

## SFT
We provide the SFT data in the `dataset` folder.
```bash
bash tasks/sft/sft.sh
```

## RL
```bash
bash scripts/train_ppo.sh
```

## Evaluate

```bash
bash scripts/eval.sh
```


## Annotation system

```bash
streamlit run annotation/annotate_page.py
```

