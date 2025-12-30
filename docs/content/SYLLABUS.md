# Prep Syllabus — Strictly Optimized

## Mental Model First (Non-negotiable)

You are preparing to be evaluated as:

> “Someone who can take a strong but imperfect large model and systematically improve it under real-world constraints.”

Every topic below maps directly to that evaluation criterion.

---

## Tier 1: Core (Must Be Excellent)

### 1. Post-Training of Large Language Models

You should be fluent enough to **design experiments**, not just explain concepts.

**What you must know**

* SFT: data quality, overfitting, curriculum effects
* Preference optimization:

  * DPO vs PPO vs GRPO (when, why, tradeoffs)
  * Implicit vs explicit reward modeling
* Stability issues in post-training
* When post-training hurts base capabilities
* Mixing multiple objectives without collapse

**Interview expectation**

* “Why did metric X improve but Y regress?”
* “How would you tune this pipeline if loss diverges after N steps?”

---

### 2. Training Dynamics & Optimization at Scale

This is where strong candidates separate.

**What you must know**

* Learning rate schedules and failure modes
* Gradient noise scale intuition
* Batch size vs convergence tradeoffs
* Regularization in large models
* Checkpoint averaging, EMA, early stopping decisions

**You should be able to**

* Diagnose training runs from logs
* Propose *minimal* interventions, not shotgun fixes

---

### 3. Evaluation & Model Debugging

This is the single highest-leverage area.

**What you must know**

* Offline evaluation design for LLMs
* Metric sensitivity and leakage
* Dataset slicing and error taxonomies
* Avoiding false positives in improvements
* Interpreting preference data

**Interview expectation**

* “Model A beats Model B on metric M. Do we ship it?”
* “How do you know this isn’t overfitting?”

---

### 4. Data as a First-Class Lever

Not data engineering — *data for training decisions*.

**What you must know**

* Data filtering and mixing strategies
* Sampling distributions
* Curriculum and replay
* Dealing with noisy / adversarial data
* Tradeoffs between data volume and data quality

---

## Tier 2: Systems for Training (Only What Matters)

You are **not** being evaluated as an infra engineer — but you must understand constraints.

### 5. Distributed Training Essentials

Focus only on what affects model behavior.

**Must understand**

* Data parallel vs model parallel (high level)
* Memory vs compute bottlenecks
* Throughput vs convergence tradeoffs
* Failure recovery and reproducibility

**Nice to have**

* Megatron-style abstractions
* ZeRO concepts (without deep implementation)

---

### 6. Efficiency as an Enabler

Only where it unlocks experimentation.

**Know conceptually**

* Mixed precision tradeoffs
* Checkpointing vs recomputation
* Why some optimizations change training dynamics

---

## Tier 3: Research Judgment (Soft but Critical)

This is often evaluated implicitly.

### 7. Experiment Design & Taste

You must demonstrate:

* Hypothesis-driven experiments
* Minimal ablations
* Knowing when *not* to run something
* How to react to negative results

### 8. Reading & Applying Research

You are not expected to invent theory, but:

* You should map papers to training knobs
* You should know when a paper will not scale

---

## What You Explicitly Do NOT Need

Do not spend prep time on:

* Serving systems
* API design
* Product analytics
* Frontend or UX
* Generic MLOps tooling
* Business KPI optimization
