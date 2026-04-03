# LLM-Course-2025-Advanced-Competition---Analysis-and-Improvements
LLM講座2025応用編最終課題コンペ

## Overview
This repository documents my approach and improvements in the final competition of the Large Language Model Course 2025 (Advanced), organized by the Matsuo-Iwasawa Laboratory at the University of Tokyo.

The competition involved multiple stages. In the initial phase, approximately 600 participants competed, from which around 200 were selected for the advanced stage within the first week.

After that, the main competition continued for three weeks, with a total of approximately 1250 participants achieving the completion criteria.

Among the advanced-stage participants (approximately 180 final submissions), I ranked 9th and was selected as an outstanding performer.

This experience reflects not only performance but also consistency in problem-solving across multiple evaluation stages.

---

## Problem
The initial model performance was limited, especially in specific domains, and simple scaling of training was not sufficient to improve results.

---

## Hypothesis
From error analysis, I observed inconsistencies and instability in model outputs across similar inputs.

Based on this, I hypothesized that data quality issues, such as duplication and distribution bias, were negatively affecting generalization performance.

---

## Approach
To validate the hypothesis, I focused on improving data quality through the following steps:

- Deduplication to reduce overfitting  
- Analysis and restructuring of data distribution  
- Targeted data augmentation for weak areas  

---

## Results
These improvements led to consistent performance gains across previously weak domains.

As a result, I achieved 9th place in the competition and was selected as an outstanding participant.

---

## Key Insights
This project highlighted that, in LLM systems, data design is as important as model architecture.

Iterative cycles of hypothesis → validation → refinement were critical to achieving improvements.

---

## Future Work
I am interested in further exploring how data design influences the behavior and performance of LLMs.

In particular, I would like to investigate:

- How differences in data distribution affect model outputs  
- How initial data conditions (e.g., bias, noise) influence robustness and behavior  
- How these effects become more complex in systems involving multiple interacting LLM agents  

Through this, I am interested in extending my understanding toward more complex environments, including LLM-based simulations.

---

## Resources

- Dataset: https://huggingface.co/datasets/kochan13/kochan13/mixed-agent-dataset-merged-clean-dedup-dbweak_2x_2
  → Processed dataset with normalization and distribution control

- Model: [（あれば）](https://huggingface.co/kochan13/lora_agentbench_qwen2.5_7b_26030106_16bit)

These were used as part of the data-centric improvement approach.

---

## Author
GitHub: https://github.com/kjboost
