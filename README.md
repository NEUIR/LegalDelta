# Legal&Delta;: Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain
## Overview
![](figs/RankCoT.png)
Legal&Delta; is a reinforcement learning framework designed to enhance legal reasoning through COT-guided information gain. During training, Legal&Delta; employs a dual-mode input setup—comprising direct answer and reasoning-augmented modes—and maximizes the information gain between them. This encourages the model to acquire meaningful reasoning patterns rather than generating superficial or redundant explanations.
Legal&Delta; follows a two-stage approach: (1) distilling latent reasoning capabilities from a powerful Large Reasoning Model (LRM), DeepSeek-R1, and (2) refining reasoning quality via differential comparisons, combined with a multidimensional reward mechanism that assesses both structural coherence and legal-domain specificity.
## Set Up
**Use `git clone` to download this project**
```
git clone https://github.com/NEUIR/RankCoT.git
cd Legal_Delta
```
**To prevent conflicts between packages, we mainly use two virtual environment management packages, one for model inference and one for model training.**

```
for model inference, please:
conda env create -n llama3_inf -f inference_environment.yml

for model training, please:
conda env create -n legal_delta -f training_environment.yml
```
