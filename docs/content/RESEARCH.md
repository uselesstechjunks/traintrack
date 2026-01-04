## 1) Uncertainty-localization:

### (A) Uncertainty estimation and calibration for LLMs

* Survey coverage of methods and failure modes: ([arXiv][1])
* “Semantic uncertainty” style work (avoid overreacting to surface form differences): ([OpenReview][2])
* Large comparative studies of uncertainty estimation / calibration across many models: ([OpenReview][3])
* Critiques arguing UQ should be evaluated in human decision-making settings, not just as a number: ([arXiv][4])

### (B) Interactive clarification under uncertainty (question-asking as a behavior)

* Example: an interactive framework that triggers question generation under high uncertainty (“pose clarifying questions”): ([OpenReview][5])

## 2) Assumption-extraction / model-adequacy: “What am I relying on, and is my representation the right kind?”

### (A) Self-reflection as a learned behavior (not necessarily weight updates)

* “Reflexion” popularized the idea of agents improving by storing reflections in memory (language-level self-feedback rather than gradient updates): ([arXiv][6])
* Work that studies *where self-reflection comes from* and how it can be modulated (“probing and modulating self-reflection”): ([arXiv][7])
* Critical evidence that “intrinsic self-correction” is limited in vanilla LMs (important negative result): ([OpenReview][8])

### (B) Retrieval + critique loops (self-critique tied to evidence)

* Self-RAG: the model learns/uses an internal policy for “should I retrieve?” + “critique/reflect” to improve factuality and quality: ([IBM Research][9])

## 3) Question-generation / experiment-selection: “What should I ask/do next to learn fastest?”

### (A) LLM-based active learning (selection or generation of informative data)

* Survey of LLM-based active learning methods and taxonomies (selection and generation): ([ACL Anthology][10])

### (B) Automatic question generation (mostly education-oriented)

* Example educational QG work (2025): ([ScienceDirect][11])

## 4) Self-improvement loops: “Use my own outputs to make myself better”

### (A) Bootstrapping reasoning data from the model itself (STaR-family)

* Original STaR (bootstrapping rationales; iterate: generate, filter by correctness, finetune, repeat): ([NeurIPS Proceedings][12])
* Variants for structured tasks like Text-to-SQL (STaR-SQL): ([ACL Anthology][13])
* “Balance exploration vs exploitation across iterations” (B-STAR): ([ICLR Proceedings][14])

### (B) Verifiable reward training (RLVR) and integrated self-verification

* Analysis of RLVR effects on reasoning and metrics (2025): ([arXiv][15])
* Example of explicitly training both solving and self-verification in one loop (“Trust, But Verify …”): ([neurips.cc][16])

### (C) Task-specific self-improvement with hindsight / replay

* CodeIt (ICML 2024): self-improvement via program sampling + hindsight relabeling + prioritized replay (for sparse-reward ARC-style problems): ([arXiv][17])

### (D) Multimodal self-improvement surveys / overviews

* A 2025 structured overview of “self-improvement in multimodal LLMs” (data collection, organization, optimization): ([arXiv][18])
* Example multimodal self-improvement via reflection / self-training: ([ACL Anthology][19])

[1]: https://arxiv.org/html/2412.05563v2 "A Survey on Uncertainty Quantification of Large Language ..."
[2]: https://openreview.net/forum?id=N4mb3MBV6J&utm_source=chatgpt.com "Improving Uncertainty Quantification in Large Language ..."
[3]: https://openreview.net/pdf?id=Q9CreVjHH7&utm_source=chatgpt.com "Revisiting Uncertainty Estimation and Calibration of Large ..."
[4]: https://arxiv.org/html/2506.07461v1 "LLM Uncertainty Quantification Should Be More Human ..."
[5]: https://openreview.net/pdf?id=nnlmcxYWlV&utm_source=chatgpt.com "Interactive Large Language Models for Reliable Answering ..."
[6]: https://arxiv.org/abs/2303.11366 "Reflexion: Language Agents with Verbal Reinforcement ..."
[7]: https://arxiv.org/html/2506.12217v1 "Probing and Modulating Self-Reflection in Language Models"
[8]: https://openreview.net/forum?id=IkmD3fKBPQ&utm_source=chatgpt.com "Large Language Models Cannot Self-Correct Reasoning Yet"
[9]: https://research.ibm.com/publications/self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection "Self-RAG: Learning to Retrieve, Generate, and Critique ..."
[10]: https://aclanthology.org/2025.acl-long.708.pdf "A Survey of LLM-based Active Learning"
[11]: https://www.sciencedirect.com/science/article/pii/S2666920X25000104 "Can large language models meet the challenge of ..."
[12]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html "STaR: Bootstrapping Reasoning With Reasoning"
[13]: https://aclanthology.org/2025.acl-long.1187.pdf "STaR-SQL: Self-Taught Reasoner for Text-to-SQL"
[14]: https://proceedings.iclr.cc/paper_files/paper/2025/file/c8db30c6f024a3f667232ed7ba5b6d47-Paper-Conference.pdf "B-STAR: MONITORING AND BALANCING EXPLORATION ..."
[15]: https://arxiv.org/abs/2506.14245 "Reinforcement Learning with Verifiable Rewards Implicitly ..."
[16]: https://neurips.cc/virtual/2025/poster/116768 "Trust, But Verify: A Self-Verification Approach ..."
[17]: https://arxiv.org/abs/2402.04858 "CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay"
[18]: https://arxiv.org/abs/2510.02665 "Self-Improvement in Multimodal Large Language Models"
[19]: https://aclanthology.org/2025.naacl-long.447.pdf "Vision-Language Models Can Self-Improve Reasoning via ..."
