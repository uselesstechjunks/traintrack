## 1) Uncertainty-localization:

### (A) Uncertainty estimation and calibration for LLMs

* Survey coverage of methods and failure modes: ([arXiv](https://arxiv.org/html/2412.05563v2))
* “Semantic uncertainty” style work (avoid overreacting to surface form differences): ([OpenReview](https://openreview.net/forum?id=N4mb3MBV6J&utm_source=chatgpt.com))
* Large comparative studies of uncertainty estimation / calibration across many models: ([OpenReview](https://openreview.net/pdf?id=Q9CreVjHH7&utm_source=chatgpt.com))
* Critiques arguing UQ should be evaluated in human decision-making settings, not just as a number: ([arXiv](https://arxiv.org/html/2506.07461v1))

### (B) Interactive clarification under uncertainty (question-asking as a behavior)

* Example: an interactive framework that triggers question generation under high uncertainty (“pose clarifying questions”): ([OpenReview](https://openreview.net/pdf?id=nnlmcxYWlV&utm_source=chatgpt.com))

## 2) Assumption-extraction / model-adequacy: “What am I relying on, and is my representation the right kind?”

### (A) Self-reflection as a learned behavior (not necessarily weight updates)

* “Reflexion” popularized the idea of agents improving by storing reflections in memory (language-level self-feedback rather than gradient updates): ([arXiv](https://arxiv.org/abs/2303.11366))
* Work that studies *where self-reflection comes from* and how it can be modulated (“probing and modulating self-reflection”): ([arXiv](https://arxiv.org/html/2506.12217v1))
* Critical evidence that “intrinsic self-correction” is limited in vanilla LMs (important negative result): ([OpenReview](https://openreview.net/forum?id=IkmD3fKBPQ&utm_source=chatgpt.com))

### (B) Retrieval + critique loops (self-critique tied to evidence)

* Self-RAG: the model learns/uses an internal policy for “should I retrieve?” + “critique/reflect” to improve factuality and quality: ([IBM Research](https://research.ibm.com/publications/self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection))

## 3) Question-generation / experiment-selection: “What should I ask/do next to learn fastest?”

### (A) LLM-based active learning (selection or generation of informative data)

* Survey of LLM-based active learning methods and taxonomies (selection and generation): ([ACL Anthology](https://aclanthology.org/2025.acl-long.708.pdf))

### (B) Automatic question generation (mostly education-oriented)

* Example educational QG work (2025): ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666920X25000104))

## 4) Self-improvement loops: “Use my own outputs to make myself better”

### (A) Bootstrapping reasoning data from the model itself (STaR-family)

* Original STaR (bootstrapping rationales; iterate: generate, filter by correctness, finetune, repeat): ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html))
* Variants for structured tasks like Text-to-SQL (STaR-SQL): ([ACL Anthology](https://aclanthology.org/2025.acl-long.1187.pdf))
* “Balance exploration vs exploitation across iterations” (B-STAR): ([ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/c8db30c6f024a3f667232ed7ba5b6d47-Paper-Conference.pdf))

### (B) Verifiable reward training (RLVR) and integrated self-verification

* Analysis of RLVR effects on reasoning and metrics (2025): ([arXiv](https://arxiv.org/abs/2506.14245))
* Example of explicitly training both solving and self-verification in one loop (“Trust, But Verify …”): ([neurips.cc](https://neurips.cc/virtual/2025/poster/116768))

### (C) Task-specific self-improvement with hindsight / replay

* CodeIt (ICML 2024): self-improvement via program sampling + hindsight relabeling + prioritized replay (for sparse-reward ARC-style problems): ([arXiv](https://arxiv.org/abs/2402.04858))

### (D) Multimodal self-improvement surveys / overviews

* A 2025 structured overview of “self-improvement in multimodal LLMs” (data collection, organization, optimization): ([arXiv](https://arxiv.org/abs/2510.02665))
* Example multimodal self-improvement via reflection / self-training: ([ACL Anthology](https://aclanthology.org/2025.naacl-long.447.pdf))
