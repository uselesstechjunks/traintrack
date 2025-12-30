# Reading List — Research Engineer / Research Scientist (Training-Focused Roles)

This readlist is structured to match the syllabus **tier-by-tier**.

---

## Tier 1 — Core Model Improvement (Non-Negotiable)

### 1. Post-Training LLMs (SFT, DPO, RLHF, Preference Optimization)

#### Primary sources (read all)

1. **RLHF Book — Nathan Lambert**
   [https://rlhfbook.com/](https://rlhfbook.com/)

2. **Hugging Face TRL Documentation**
   [https://huggingface.co/docs/trl/main/en/index](https://huggingface.co/docs/trl/main/en/index)

3. **Hugging Face RLHF Guide**
   [https://huggingface.co/learn/deep-rl-course/en/unitbonus3/rlhf](https://huggingface.co/learn/deep-rl-course/en/unitbonus3/rlhf)

5. **Direct Preference Optimization (DPO)**
   [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

6. **Group Relative Policy Optimization (GRPO)**
   [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

7. **OpenAI — Learning from Human Preferences**
   [https://openai.com/research/learning-from-human-preferences](https://openai.com/research/learning-from-human-preferences)

---

#### Supporting intuition (highly recommended)

1. **Sebastian Raschka — LLM Finetuning Overview**
   [https://magazine.sebastianraschka.com/p/finetuning-large-language-models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)

2. **Lilian Weng — RLHF**
   [https://lilianweng.github.io/tags/rlhf/](https://lilianweng.github.io/tags/rlhf/)

---

### 2. Training Dynamics & Optimization at Scale

#### Core reading

1. **Scaling Laws for Neural Language Models**
   [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

2. **Chinchilla Scaling Laws**
   [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

3. **Gradient Noise Scale**
   [https://arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)

4. **On Large-Batch Training**
   [https://arxiv.org/abs/1609.04836](https://arxiv.org/abs/1609.04836)

5. **Andrej Karpathy — Neural Nets: 20 Years**
   [https://karpathy.github.io/2022/03/14/lecun1989/](https://karpathy.github.io/2022/03/14/lecun1989/)

---

#### Practitioner-level intuition

1. **OpenAI — Scaling Laws Blog**
   [https://openai.com/research/scaling-laws-for-neural-language-models](https://openai.com/research/scaling-laws-for-neural-language-models)

2. **Stanford CS25 (LLMs)**
   [https://web.stanford.edu/class/cs25/](https://web.stanford.edu/class/cs25/)

---

### 3. Evaluation, Metrics, and Debugging

#### Must-read

1. **OpenAI — Evaluating Language Models**
   [https://openai.com/research/evaluating-language-models](https://openai.com/research/evaluating-language-models)

2. **Holistic Evaluation of Language Models**
   [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)

3. **Google — Beyond Accuracy**
   [https://arxiv.org/abs/2506.04723](https://arxiv.org/abs/2506.04723)

---

#### Practical evaluation intuition

1. **Anthropic — Red Teaming Language Models**
   [https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned](https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned)

---

### 4. Data as a Training Lever

#### Core sources

1. **Data Selection for Language Models**
   [https://arxiv.org/abs/2402.16827](https://arxiv.org/abs/2402.16827)

2. **The Pile**
   [https://arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027)

---

#### Practitioner insight

1. **OpenAI Cookbook — Data Preparation**
   [https://cookbook.openai.com/examples/chat_finetuning_data_prep](https://cookbook.openai.com/examples/chat_finetuning_data_prep)

2. **Training Data Cascades**
   [https://arxiv.org/abs/2209.07753](https://arxiv.org/abs/2209.07753)
   Why: How early data decisions poison downstream stages.

---

## Tier 2 — Systems for Training (Only What Affects Models)

### 5. Distributed Training

1. **Megatron-LM**
   [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
   Why: Why model parallelism exists.

2. **ZeRO**
   [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
   Why: Memory vs compute tradeoffs.

3. **Hugging Face — Parallelism Guide**
   [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)
   Why: Clear conceptual mapping of DP / MP / PP.

---

### 6. Efficiency & Precision

1. **Mixed Precision Training**
   [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
   Why: Precision changes optimization dynamics.

2. **PyTorch Activation Checkpointing**
   [https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html)
   Why: Memory tradeoffs and hidden costs.

---

## Tier 3 — Research Judgment & Taste

### 7. Experiment Design

1. **Karpathy — A Recipe for Training Neural Networks**
   [https://karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)
   Why: Research taste distilled.

2. **Richard Sutton — The Bitter Lesson**
   [http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
   Why: Scaling vs cleverness.

---

### 8. Reading Research Effectively

1. **How to Read ML Papers — Sanjeev Arora**
   [https://www.cs.princeton.edu/~arora/pubs/reading.pdf](https://www.cs.princeton.edu/~arora/pubs/reading.pdf)
   Why: Extract actionable knobs.

2. **Meta AI — Importance of Experimentation**
   [https://ai.facebook.com/blog/the-importance-of-experimentation-in-ml-research/](https://ai.facebook.com/blog/the-importance-of-experimentation-in-ml-research/)
   Why: Industry research reality.

---

## Tier 4 — Failure, Drift, and Emergent Behavior

### Targeted Readings

**Objective Misalignment**

* Reward Is Not the Optimization Objective
  [https://www.lesswrong.com/posts/bmfnzqTjQ9H6k8YyZ/reward-is-not-the-optimization-objective](https://www.lesswrong.com/posts/bmfnzqTjQ9H6k8YyZ/reward-is-not-the-optimization-objective)
* Specification Gaming (DeepMind)
  [https://deepmind.google/discover/blog/specification-gaming-the-flawed-way-to-design-ai/](https://deepmind.google/discover/blog/specification-gaming-the-flawed-way-to-design-ai/)
* Faulty Reward Functions (OpenAI)
  [https://openai.com/research/faulty-reward-functions](https://openai.com/research/faulty-reward-functions)

**Continual Training**

* Catastrophic Forgetting
  [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)
* Risks from Learned Optimization (Anthropic)
  [https://www.anthropic.com/research/risks-from-learned-optimization](https://www.anthropic.com/research/risks-from-learned-optimization)

**Causal Reasoning**

* A/B Testing ML Systems
  [https://research.google/pubs/pub46753/](https://research.google/pubs/pub46753/)
* Causal Confusion in ML
  [https://arxiv.org/abs/1806.09305](https://arxiv.org/abs/1806.09305)

**Human Feedback Pathologies**

* Social Biases of RLHF
  [https://arxiv.org/abs/2303.17548](https://arxiv.org/abs/2303.17548)
* Challenges in Data Annotation
  [https://openai.com/research/challenges-in-data-annotation](https://openai.com/research/challenges-in-data-annotation)

**Drift**

* Why Models Fail After Deployment
  [https://arxiv.org/abs/2302.00942](https://arxiv.org/abs/2302.00942)
* Evaluation Is Broken
  [https://www.alignmentforum.org/posts/9iA6WZp4bKc9bZ8F8/evaluation-is-broken](https://www.alignmentforum.org/posts/9iA6WZp4bKc9bZ8F8/evaluation-is-broken)

---

### Case Studies

* Microsoft Tay Postmortem
  [https://www.microsoft.com/en-us/research/blog/learning-from-tay/](https://www.microsoft.com/en-us/research/blog/learning-from-tay/)
* GPT-4 System Card
  [https://cdn.openai.com/papers/gpt-4-system-card.pdf](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
* Anthropic — Constitutional AI
  [https://www.anthropic.com/research/constitutional-ai](https://www.anthropic.com/research/constitutional-ai)
* InstructGPT
  [https://openai.com/research/instruction-following](https://openai.com/research/instruction-following)

---

## Tier 5 — Architecture, Safety, and Strategic Judgment

### Architecture

* Attention Is All You Need
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
* Foundation Models Risks
  [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)
* Transformer Circuits
  [https://transformer-circuits.pub/](https://transformer-circuits.pub/)

### Safety–Capability Tradeoffs

* Helpful, Harmless, Honest
  [https://www.anthropic.com/research/helpful-harmless-honest](https://www.anthropic.com/research/helpful-harmless-honest)
* Measuring Harms and Benefits
  [https://arxiv.org/abs/2303.16248](https://arxiv.org/abs/2303.16248)
* GPT-4 System Card (re-read)

### Compute & Strategy

* Chinchilla Revisited
  [https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training](https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training)
* GPT-3 Lessons
  [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
* Carbon Cost of ML
  [https://arxiv.org/abs/1906.02243](https://arxiv.org/abs/1906.02243)

---

## Tier 6 — Pretraining Context (Targeted)

* Tokenization and data mixture papers (survey-level)
* Pretraining objective design (causal effects on post-training)
* Limits of post-training compensation

(Use GPT-3, Chinchilla, and Pile papers as anchors.)

---

## Tier 7 — Execution & Proof of Skill

### Evaluation Harnesses

* OpenAI Evals
* HELM (Stanford)
* Prompt sensitivity and variance analyses

### Runbooks

* Training collapse diagnosis
* Reward hacking detection
* Metric disagreement resolution

(Primary learning via practice + postmortems.)

---

## How to Use This

For each tier:

1. Read one **core** source
2. One **practitioner** source
3. Write:

   * What knob this gives you
   * When it fails
   * What metric it affects
