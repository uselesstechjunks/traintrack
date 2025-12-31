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

1. **OpenAI Evals (framework and docs)**
   [https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals](https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals)

2. **Process for Adapting Language Models to Society (PALMS)**
   [https://cdn.openai.com/palms.pdf](https://cdn.openai.com/palms.pdf)​

3. **Measuring short-form factuality in large language models (SimpleQA)**
   [https://cdn.openai.com/papers/simpleqa.pdf](https://cdn.openai.com/papers/simpleqa.pdf)​

4. **Holistic Evaluation of Language Models**
   [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)

5. **Google — Beyond Accuracy**
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
   [https://storage.googleapis.com/gweb-research2023-media/pubtools/5936.pdf](https://storage.googleapis.com/gweb-research2023-media/pubtools/5936.pdf)

---

## Tier 2 — Systems for Training (Only What Affects Models)

### 5. Distributed Training

1. **Megatron-LM**
   [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)

2. **ZeRO**
   [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)

3. **Hugging Face — Parallelism Guide**
   [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)

---

### 6. Efficiency & Precision

1. **Mixed Precision Training**
   [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)

2. **PyTorch Activation Checkpointing**
   [https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html)

---

## Tier 3 — Research Judgment & Taste

### 7. Experiment Design

1. **Karpathy — A Recipe for Training Neural Networks**
   [https://karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)

2. **Richard Sutton — The Bitter Lesson**
   [http://www.incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

---

### 8. Reading Research Effectively

1. **How to Read ML Papers — NVIDIA**
   [https://developer.nvidia.com/blog/how-to-read-research-papers-a-pragmatic-approach-for-ml-practitioners/](https://developer.nvidia.com/blog/how-to-read-research-papers-a-pragmatic-approach-for-ml-practitioners/)

2. **Importance of Experimentation**
   [https://www.chrishayduk.com/p/the-role-of-experimentation-in-fundamental](https://www.chrishayduk.com/p/the-role-of-experimentation-in-fundamental)

---

## Tier 4 — Failure, Drift, and Emergent Behavior

### Targeted Readings

**Objective Misalignment**

* Reward Is Not the Optimization Objective
  [https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target](https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target)
* Specification Gaming (DeepMind)
  [https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
* Faulty Reward Functions (OpenAI)
  [https://openai.com/research/faulty-reward-functions](https://openai.com/research/faulty-reward-functions)

**Continual Training**

* Catastrophic Forgetting
  [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)
* Risks from Learned Optimization
  [https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB](https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB)

**Causal Reasoning**

* A/B Testing ML Systems
  [https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36500.pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36500.pdf)
* Causal Confusion in ML
  [https://arxiv.org/abs/1905.11979](https://arxiv.org/abs/1905.11979)

**Human Feedback Pathologies**

* Social Biases of RLHF
  [https://arxiv.org/abs/2303.17548](https://arxiv.org/abs/2303.17548)
* Challenges in Data Annotation
  [https://hai.stanford.edu/news/exploring-complex-ethical-challenges-data-annotation](https://hai.stanford.edu/news/exploring-complex-ethical-challenges-data-annotation)

**Drift**

* Why Models Fail After Deployment
  [https://arxiv.org/abs/2203.11070](https://arxiv.org/abs/2203.11070)
* Evaluation Is Broken
  [https://www.alignmentforum.org/posts/dBmfb76zx6wjPsBC7](https://www.alignmentforum.org/posts/dBmfb76zx6wjPsBC7)

---

### Case Studies

* Microsoft Tay Postmortem
  [https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/](https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/)
* GPT-4 System Card
  [https://cdn.openai.com/papers/gpt-4-system-card.pdf](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
* Anthropic — Constitutional AI
  [https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
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
  [https://arxiv.org/pdf/2204.05862](https://arxiv.org/pdf/2204.05862)
* Measuring Harms and Benefits
  [https://arxiv.org/abs/2304.04359](https://arxiv.org/abs/2304.04359)

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
