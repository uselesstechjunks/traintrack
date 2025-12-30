# Tier 1 — Non-Negotiable Core

## 1. Post-Training LLMs (SFT, DPO, RLHF, preference optimization)

### Primary sources (read all)

**Hugging Face RLHF Book**
[https://huggingface.co/docs/trl/main/en/rlhf](https://huggingface.co/docs/trl/main/en/rlhf)
Why: Best single practical overview of modern post-training pipelines. Clear, code-adjacent, and realistic.

**DPO paper (Direct Preference Optimization)**
[https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
Why: You must understand *why* DPO works and when it breaks. Focus on the objective, not proofs.

**GRPO (Group Relative Policy Optimization)**
[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
Why: This is now appearing in real systems. You should understand the intuition and stability tradeoffs.

**OpenAI blog: Learning from Human Preferences**
[https://openai.com/research/learning-from-human-preferences](https://openai.com/research/learning-from-human-preferences)
Why: Foundational intuition for preference-based learning and reward modeling.

---

### Supporting intuition (highly recommended)

**Sebastian Raschka – LLM finetuning overview**
[https://sebastianraschka.com/blog/2023/llm-finetuning.html](https://sebastianraschka.com/blog/2023/llm-finetuning.html)
Why: Clear mental model for SFT vs post-training vs instruction tuning.

**Lil’Log (Lilian Weng) – RLHF**
[https://lilianweng.github.io/posts/2023-03-15-rlhf/](https://lilianweng.github.io/posts/2023-03-15-rlhf/)
Why: Extremely clean conceptual grounding. Read once, revisit often.

---

## 2. Training Dynamics & Optimization at Scale

### Core reading

**Andrej Karpathy – Deep Neural Nets: 20 Years**
[https://karpathy.github.io/2022/12/12/nn-explainer/](https://karpathy.github.io/2022/12/12/nn-explainer/)
Why: Surprisingly relevant intuition for modern large-scale training behavior.

**Gradient Noise Scale (McCandlish et al.)**
[https://arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)
Why: This paper explains batch size vs convergence better than any blog post.

**On Large-Batch Training**
[https://arxiv.org/abs/1609.04836](https://arxiv.org/abs/1609.04836)
Why: You should know why large batches fail and when they don’t.

---

### Practitioner-level intuition

**OpenAI – Tricks of Stable Large-Scale Training**
[https://openai.com/research/scaling-laws-for-neural-language-models](https://openai.com/research/scaling-laws-for-neural-language-models)
Why: Scaling laws + training stability intuition. Read for patterns, not equations.

**Stanford CS25 lecture notes (LLMs training)**
[https://web.stanford.edu/class/cs25/](https://web.stanford.edu/class/cs25/)
Why: Real-world training insights, not just theory.

---

## 3. Evaluation, Metrics, and Model Debugging

This is the **highest ROI area**.

### Must-read

**OpenAI – Evaluating Language Models**
[https://openai.com/research/evaluating-language-models](https://openai.com/research/evaluating-language-models)
Why: Ground truth on why LLM evaluation is hard.

**Holistic Evaluation of Language Models**
[https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)
Why: Teaches you to think in slices, not scalar metrics.

**Google – Beyond Accuracy**
[https://research.google/blog/beyond-accuracy-behavioral-testing-of-nlp-models/](https://research.google/blog/beyond-accuracy-behavioral-testing-of-nlp-models/)
Why: Shows how strong teams think about failures.

---

### Practical evaluation intuition

**Lil’Log – LLM Evaluation**
[https://lilianweng.github.io/posts/2024-02-05-llm-evaluation/](https://lilianweng.github.io/posts/2024-02-05-llm-evaluation/)
Why: Practical taxonomy of evaluation methods and their pitfalls.

**Anthropic – Red Teaming Language Models**
[https://www.anthropic.com/research/red-teaming-language-models](https://www.anthropic.com/research/red-teaming-language-models)
Why: Demonstrates structured error discovery.

---

## 4. Data as a Lever (Filtering, Mixing, Curriculum)

### Core sources

**Data Selection for Language Models**
[https://arxiv.org/abs/2202.08906](https://arxiv.org/abs/2202.08906)
Why: Shows that data quality and mixing matter more than people think.

**The Pile paper**
[https://arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027)
Why: Data curation philosophy at scale.

---

### Practitioner insight

**OpenAI Cookbook – Data curation**
[https://cookbook.openai.com/examples/data_preparation](https://cookbook.openai.com/examples/data_preparation)
Why: Practical heuristics, not academic abstractions.

**Google – Training Data Cascades**
[https://arxiv.org/abs/2209.07753](https://arxiv.org/abs/2209.07753)
Why: Teaches you how bad data decisions propagate downstream.

---

# Tier 2 — Systems for Training (Only What Matters)

## 5. Distributed Training (Conceptual, not infra-heavy)

### Core understanding

**Megatron-LM paper**
[https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
Why: You should understand *why* model parallelism exists.

**ZeRO paper**
[https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
Why: Memory vs compute tradeoffs explained cleanly.

---

### Practical summaries

**Hugging Face – Parallelism guide**
[https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)
Why: Clear explanation of DP, MP, PP without noise.

---

## 6. Efficiency That Affects Training Behavior

**Mixed Precision Training (Micikevicius et al.)**
[https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
Why: Know when precision changes optimization dynamics.

**Activation Checkpointing**
[https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html)
Why: Practical memory tradeoffs and their costs.

---

# Tier 3 — Research Judgment & Taste

This is subtle but decisive.

## 7. Experiment Design & Taste

**Andrej Karpathy – Research Taste**
[https://karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)
Why: This is explicitly evaluated in interviews, even if unstated.

**Richard Sutton – Bitter Lesson**
[http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
Why: Teaches long-term thinking about scaling vs cleverness.

---

## 8. Reading Research Without Getting Lost

**How to Read ML Papers (Sanjeev Arora)**
[https://www.cs.princeton.edu/~arora/pubs/reading.pdf](https://www.cs.princeton.edu/~arora/pubs/reading.pdf)
Why: Helps you map papers to knobs you can actually turn.

**Meta AI – On empirical ML research**
[https://ai.facebook.com/blog/the-importance-of-experimentation-in-ml-research/](https://ai.facebook.com/blog/the-importance-of-experimentation-in-ml-research/)
Why: Emphasizes experimentation over novelty.

---

# What You Can Safely Skip for Now

* General MLOps books
* Product analytics blogs
* Kaggle-style content
* End-to-end deployment tutorials
* Generic “LLM from scratch” courses

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
