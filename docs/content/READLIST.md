# Tier 1 — Non-Negotiable Core

## 1. Post-Training LLMs (SFT, DPO, RLHF, preference optimization)

### Primary sources (read all)

1. **RLHF Book (Nathan Lambert)**  
   [https://rlhfbook.com/](https://rlhfbook.com/)  
   Why: Complete practitioner-oriented synthesis of RLHF, preference learning, and post-training; ties objectives, datasets, infra, and evaluation together.
2. **Hugging Face TRL Documentation (RLHF & DPO)**  
   [https://huggingface.co/docs/trl/main/en/index](https://huggingface.co/docs/trl/main/en/index)  
   Why: Code-adjacent view of modern post-training pipelines; shows how ideas become systems.
3. **Hugging Face RLHF Book**  
   [https://huggingface.co/docs/trl/main/en/rlhf](https://huggingface.co/docs/trl/main/en/rlhf)  
   Why: Best single practical overview of modern post-training pipelines. Clear, code-adjacent, and realistic.
4. **DPO paper (Direct Preference Optimization)**  
   [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)  
   Why: You must understand *why* DPO works and when it breaks. Focus on the objective, not proofs.
5. **GRPO (Group Relative Policy Optimization)**  
   [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)  
   Why: This is now appearing in real systems. You should understand the intuition and stability tradeoffs.
6. **OpenAI blog: Learning from Human Preferences**  
   [https://openai.com/research/learning-from-human-preferences](https://openai.com/research/learning-from-human-preferences)  
   Why: Foundational intuition for preference-based learning and reward modeling.

---

### Supporting intuition (highly recommended)

1. **Sebastian Raschka – LLM finetuning overview**  
   [https://sebastianraschka.com/blog/2023/llm-finetuning.html](https://sebastianraschka.com/blog/2023/llm-finetuning.html)  
   Why: Clear mental model for SFT vs post-training vs instruction tuning.
2. **Lil’Log (Lilian Weng) – RLHF**  
   [https://lilianweng.github.io/posts/2023-03-15-rlhf/](https://lilianweng.github.io/posts/2023-03-15-rlhf/)  
   Why: Extremely clean conceptual grounding. Read once, revisit often.

---

## 2. Training Dynamics & Optimization at Scale

### Core reading

1. **Scaling Laws for Neural Language Models (Kaplan et al.)**  
   [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)  
   Why: Understand what truly scales (and what does not) to guide training strategy.
2. **Chinchilla Scaling Laws (Hoffmann et al.)**  
   [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)  
   Why: Data vs compute tradeoffs that directly influence training budget decisions.
3. **Andrej Karpathy – Deep Neural Nets: 20 Years**  
   [https://karpathy.github.io/2022/12/12/nn-explainer/](https://karpathy.github.io/2022/12/12/nn-explainer/)  
   Why: Surprisingly relevant intuition for modern large-scale training behavior.
4. **Gradient Noise Scale (McCandlish et al.)**  
   [https://arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)  
   Why: This paper explains batch size vs convergence better than any blog post.
5. **On Large-Batch Training**  
   [https://arxiv.org/abs/1609.04836](https://arxiv.org/abs/1609.04836)  
   Why: You should know why large batches fail and when they don’t.

---

### Practitioner-level intuition

1. **OpenAI – Tricks of Stable Large-Scale Training**  
   [https://openai.com/research/scaling-laws-for-neural-language-models](https://openai.com/research/scaling-laws-for-neural-language-models)  
   Why: Scaling laws + training stability intuition. Read for patterns, not equations.
2. **Stanford CS25 lecture notes (LLMs training)**  
   [https://web.stanford.edu/class/cs25/](https://web.stanford.edu/class/cs25/)  
   Why: Real-world training insights, not just theory.

---

## 3. Evaluation, Metrics, and Model Debugging

This is the **highest ROI area**.

### Must-read

1. **OpenAI – Evaluating Language Models**  
   [https://openai.com/research/evaluating-language-models](https://openai.com/research/evaluating-language-models)  
   Why: Ground truth on why LLM evaluation is hard.
2. **Holistic Evaluation of Language Models**  
   [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)  
   Why: Teaches you to think in slices, not scalar metrics.
3. **Google – Beyond Accuracy**  
   [https://research.google/blog/beyond-accuracy-behavioral-testing-of-nlp-models/](https://research.google/blog/beyond-accuracy-behavioral-testing-of-nlp-models/)  
   Why: Shows how strong teams think about failures.

---

### Practical evaluation intuition

1. **Lil’Log – LLM Evaluation**  
   [https://lilianweng.github.io/posts/2024-02-05-llm-evaluation/](https://lilianweng.github.io/posts/2024-02-05-llm-evaluation/)  
   Why: Practical taxonomy of evaluation methods and their pitfalls.
2. **Anthropic – Red Teaming Language Models**  
   [https://www.anthropic.com/research/red-teaming-language-models](https://www.anthropic.com/research/red-teaming-language-models)  
   Why: Demonstrates structured error discovery.

---

## 4. Data as a Lever (Filtering, Mixing, Curriculum)

### Core sources

1. **Data Selection for Language Models**  
   [https://arxiv.org/abs/2202.08906](https://arxiv.org/abs/2202.08906)  
   Why: Shows that data quality and mixing matter more than people think.
2. **The Pile paper**  
   [https://arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027)  
   Why: Data curation philosophy at scale.

---

### Practitioner insight

1. **OpenAI Cookbook – Data curation**  
   [https://cookbook.openai.com/examples/data_preparation](https://cookbook.openai.com/examples/data_preparation)  
   Why: Practical heuristics, not academic abstractions.
2. **Google – Training Data Cascades**  
   [https://arxiv.org/abs/2209.07753](https://arxiv.org/abs/2209.07753)  
   Why: Teaches you how bad data decisions propagate downstream.

---

# Tier 2 — Systems for Training (Only What Matters)

## 5. Distributed Training (Conceptual, not infra-heavy)

### Core understanding

1. **Megatron-LM paper**  
   [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)  
   Why: You should understand *why* model parallelism exists.
2. **ZeRO paper**  
   [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)  
   Why: Memory vs compute tradeoffs explained cleanly.

---

### Practical summaries

1. **Hugging Face – Parallelism guide**  
   [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)  
   Why: Clear explanation of DP, MP, PP without noise.

---

## 6. Efficiency That Affects Training Behavior

1. **Mixed Precision Training (Micikevicius et al.)**  
   [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)  
   Why: Know when precision changes optimization dynamics.
2. **Activation Checkpointing**  
   [https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html)  
   Why: Practical memory tradeoffs and their costs.

---

# Tier 3 — Research Judgment & Taste

This is subtle but decisive.

## 7. Experiment Design & Taste

1. **Andrej Karpathy – Research Taste**  
   [https://karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)  
   Why: This is explicitly evaluated in interviews, even if unstated.
2. **Richard Sutton – Bitter Lesson**  
   [http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)  
   Why: Teaches long-term thinking about scaling vs cleverness.

---

## 8. Reading Research Without Getting Lost

1. **How to Read ML Papers (Sanjeev Arora)**  
   [https://www.cs.princeton.edu/~arora/pubs/reading.pdf](https://www.cs.princeton.edu/~arora/pubs/reading.pdf)  
   Why: Helps you map papers to knobs you can actually turn.
2. **Meta AI – On empirical ML research**  
   [https://ai.facebook.com/blog/the-importance-of-experimentation-in-ml-research/](https://ai.facebook.com/blog/the-importance-of-experimentation-in-ml-research/)  
   Why: Emphasizes experimentation over novelty.

---

# Frontier / Ongoing Discourse (Optional, high-ceiling)

1. **Alignment Forum – RLHF / Preference Learning**  
   [https://www.alignmentforum.org/](https://www.alignmentforum.org/)  
   Why: Many cutting-edge RLHF discussions appear here first.
2. **LessWrong – LLM Training & Alignment**  
   [https://www.lesswrong.com/tag/large-language-models](https://www.lesswrong.com/tag/large-language-models)  
   Why: Useful for stress-testing ideas (read skeptically).
3. **r/MachineLearning (selected threads)**  
   [https://www.reddit.com/r/MachineLearning/](https://www.reddit.com/r/MachineLearning/)  
   Why: Occasional high-quality postmortems and paper discussions; filter out noise.

---

# READLIST — Tier 4: Failure, Drift, and Emergent Behavior

## A. Targeted Readings (Conceptual + Practitioner)

These are **surgical** additions. Do not add many more.

### Objective Misalignment & Emergent Behavior

1. **Reward Is Not the Optimization Objective**  
   [https://www.lesswrong.com/posts/bmfnzqTjQ9H6k8YyZ/reward-is-not-the-optimization-objective](https://www.lesswrong.com/posts/bmfnzqTjQ9H6k8YyZ/reward-is-not-the-optimization-objective)  
   Why: Canonical explanation of why proxy objectives break. Essential mental model.
2. **Specification Gaming Examples (DeepMind)**  
   [https://deepmind.google/discover/blog/specification-gaming-the-flawed-way-to-design-ai/](https://deepmind.google/discover/blog/specification-gaming-the-flawed-way-to-design-ai/)  
   Why: Concrete examples of objective exploitation.
3. **OpenAI – Faulty Reward Functions**  
   [https://openai.com/research/faulty-reward-functions](https://openai.com/research/faulty-reward-functions)  
   Why: Real-world failures from misaligned rewards.

---

### Continual Training & Forgetting

1. **On Catastrophic Forgetting**  
   [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)  
   Why: Foundational paper; concepts apply directly to iterative LLM post-training.
2. **Anthropic – Model Editing and Fine-Tuning Risks**  
   [https://www.anthropic.com/research/risks-from-learned-optimization](https://www.anthropic.com/research/risks-from-learned-optimization)  
   Why: Shows how repeated interventions can create hidden risks.

---

### Causal Reasoning in ML Experiments

1. **The Problem with A/B Testing ML Systems**  
   [https://research.google/pubs/pub46753/](https://research.google/pubs/pub46753/)  
   Why: Highlights confounders and false conclusions in ML experiments.
2. **Causal Confusion in Machine Learning**  
   [https://arxiv.org/abs/1806.09305](https://arxiv.org/abs/1806.09305)  
   Why: Clear articulation of why ML improvements are hard to attribute.

---

### Human-in-the-Loop Pathologies

1. **The Social Biases of RLHF**  
   [https://arxiv.org/abs/2303.17548](https://arxiv.org/abs/2303.17548)  
   Why: How human preferences distort model behavior.
2. **OpenAI – Challenges in Data Annotation**  
   [https://openai.com/research/challenges-in-data-annotation](https://openai.com/research/challenges-in-data-annotation)  
   Why: Practical annotation failure modes.

---

### Drift & Long-Term Degradation

1. **Why Models Fail After Deployment**  
   [https://arxiv.org/abs/2302.00942](https://arxiv.org/abs/2302.00942)  
   Why: Realistic view of post-deployment degradation.
2. **Evaluation Is Broken**  
   [https://www.alignmentforum.org/posts/9iA6WZp4bKc9bZ8F8/evaluation-is-broken](https://www.alignmentforum.org/posts/9iA6WZp4bKc9bZ8F8/evaluation-is-broken)  
   Why: Why metrics silently lose meaning over time.

---

## B. Case Studies & Post-Mortems (Read Slowly)

These are **gold**. Read for reasoning, not conclusions.

### Real Failures

1. **Microsoft Tay Chatbot Postmortem**  
   [https://www.microsoft.com/en-us/research/blog/learning-from-tay/](https://www.microsoft.com/en-us/research/blog/learning-from-tay/)  
   Why: Classic human-feedback failure.
2. **OpenAI GPT-4 System Card**  
   [https://cdn.openai.com/papers/gpt-4-system-card.pdf](https://cdn.openai.com/papers/gpt-4-system-card.pdf)  
   Why: Concrete discussion of risks, mitigations, and tradeoffs.
3. **Anthropic – Claude Safety Iterations**  
   [https://www.anthropic.com/research/constitutional-ai](https://www.anthropic.com/research/constitutional-ai)  
   Why: Iterative post-training tradeoffs and failure mitigation.

---

### Emergent Behavior & Drift

1. **InstructGPT Failure Modes**  
   [https://openai.com/research/instruction-following](https://openai.com/research/instruction-following)  
   Why: Shows how instruction tuning reshapes behavior.
2. **Alignment Forum – Emergent Misalignment Threads**  
   [https://www.alignmentforum.org/tag/emergent-behavior](https://www.alignmentforum.org/tag/emergent-behavior)  
   Why: Frontier discussions on unexpected behavior.

---

# What You Can Safely Skip for Now

* General MLOps books
* Product analytics blogs
* Kaggle-style content
* End-to-end deployment tutorials
* Generic “LLM from scratch” courses

---

# Tier 5 — Architecture, Safety, and Strategic Judgment

## A. Targeted Readings (Senior-Level)

### Architectural Bias & Limits

1. **Attention Is All You Need**  
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)  
   Why: Not for novelty — for understanding the inductive bias you are implicitly relying on.
2. **On the Opportunities and Risks of Foundation Models**  
   [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)  
   Why: Frames architectural choices in terms of emergent behavior and limits.
3. **Transformer Circuits (Anthropic)**  
   [https://transformer-circuits.pub/](https://transformer-circuits.pub/)  
   Why: Helps reason about what architectures encode vs what training fixes.

---

### Safety–Capability Tradeoffs

1. **OpenAI – GPT-4 System Card**  
   [https://cdn.openai.com/papers/gpt-4-system-card.pdf](https://cdn.openai.com/papers/gpt-4-system-card.pdf)  
   Why: Explicit discussion of capability loss vs safety gains.
2. **Anthropic – Helpful, Harmless, Honest**  
   [https://www.anthropic.com/research/helpful-harmless-honest](https://www.anthropic.com/research/helpful-harmless-honest)  
   Why: The original framing of safety–capability tension.
3. **Measuring Harms and Benefits of LLMs**  
   [https://arxiv.org/abs/2303.16248](https://arxiv.org/abs/2303.16248)  
   Why: Concrete attempts to quantify tradeoffs.

---

### Compute & Scaling Decisions

1. **Chinchilla Revisited (DeepMind blog)**  
   [https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training](https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training)  
   Why: Shows how compute constraints shape *strategy*, not just models.
2. **The Bitter Lesson (re-read here)**  
   [http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)  
   Why: Recontextualize it now that you understand constraints.

---

### Strategic ML Judgment

1. **What We Learned from Training GPT-3**  
   [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)  
   Why: Practical lessons about scale, cost, and failure.
2. **Meta – Large-Scale ML Lessons**  
   [https://ai.facebook.com/blog/large-scale-machine-learning-at-facebook/](https://ai.facebook.com/blog/large-scale-machine-learning-at-facebook/)  
   Why: How real teams balance quality, cost, and velocity.

---

## B. Case Studies & Decision Narratives

### Architecture & Scale

1. **Why Bigger Models Aren’t Always Better**  
   [https://www.lesswrong.com/posts/Y2vZJg5xK2J4tqF3H/why-bigger-models-aren-t-always-better](https://www.lesswrong.com/posts/Y2vZJg5xK2J4tqF3H/why-bigger-models-aren-t-always-better)  
   Why: Realistic limits of scaling.

---

### Safety vs Utility

1. **Anthropic Claude Refusal Analysis**  
   [https://www.anthropic.com/research/assessing-constitutional-ai](https://www.anthropic.com/research/assessing-constitutional-ai)  
   Why: Shows practical refusal tradeoffs.

---

### Compute Constraints

1. **Google – Carbon Cost of ML**  
   [https://arxiv.org/abs/1906.02243](https://arxiv.org/abs/1906.02243)  
   Why: Forces disciplined thinking about compute as a finite resource.

---

# How to Use This (Very Important)

Do **not** read everything linearly.

For each topic:

1. Read one **core** source
2. Cross-check intuition with one **practitioner** source
3. Write a 5–10 line summary:

   * What knob does this give me?
   * When does it fail?
   * What metric does it affect?
