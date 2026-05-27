# Prep Syllabus — Research Engineer / Research Scientist (LLMs & Large-Scale Training)

## Mental Model (Non-Negotiable)

You are preparing to be evaluated as:

> **Someone who can take a strong but imperfect large model and systematically improve it under real-world constraints (data, compute, safety, iteration speed).**

Every topic below exists because it shows up:

* in senior interviews, or
* in day-to-day work on post-training / model quality teams.

---

## Tier 0: Foundations You Are Assumed to Have (No Over-Indexing)

You are assumed to already be comfortable with:

* Backpropagation, SGD variants, basic optimization
* Transformer architecture at a high level
* PyTorch/JAX usage for model training
* Probability, linear algebra, basic statistics

You **do not** need to re-study these unless gaps surface.

---

## Tier 1: Core Model Improvement Skills (Must Be Excellent)

### 1. Post-Training of Large Language Models

You should be fluent enough to **design and debug experiments**, not just explain concepts.

**What you must know**

* SFT: data quality, overfitting, curriculum effects
* Preference optimization:

  * DPO vs PPO vs GRPO (when, why, tradeoffs)
  * Implicit vs explicit reward modeling
* Stability issues in post-training
* When post-training hurts base capabilities
* Mixing multiple objectives without collapse

**Expected capability**

* Design post-training pipelines
* Diagnose collapse, reward hacking, regressions

---

### 2. Training Dynamics & Optimization at Scale

This is where strong candidates separate.

**What you must know**

* Learning-rate schedules and failure modes
* Gradient noise scale intuition
* Batch size vs convergence tradeoffs
* Regularization in large models
* EMA, checkpoint averaging, early stopping

**You should be able to**

* Read training curves and logs causally
* Propose **minimal**, targeted interventions

---

### 3. Evaluation & Model Debugging

This is the **highest-leverage skill**.

**What you must know**

* Offline evaluation design for LLMs
* Metric sensitivity, leakage, proxy failure
* Dataset slicing and error taxonomies
* Interpreting preference and human feedback data

**Expected capability**

* Decide whether improvements are real
* Explain *why* a metric moved (or didn’t)

---

### 4. Data as a First-Class Training Lever

Not data engineering — *data for model behavior*.

**What you must know**

* Filtering vs reweighting vs mixing
* Sampling distributions and curricula
* Replay, freshness, and data decay
* Noisy, adversarial, and biased data
* Data volume vs data quality tradeoffs

---

## Tier 2: Systems for Training (Only What Affects Model Behavior)

You are **not** an infra engineer, but you must reason under constraints.

### 5. Distributed Training Essentials

**Must understand**

* Data vs model vs pipeline parallelism (conceptually)
* Memory vs compute bottlenecks
* Throughput vs convergence tradeoffs
* Failure recovery and reproducibility

**Nice to have**

* Megatron-style abstractions
* ZeRO-style memory optimizations

---

### 6. Efficiency as an Enabler (Not a Goal)

**Know conceptually**

* Mixed precision tradeoffs
* Checkpointing vs recomputation
* Why some system optimizations change optimization dynamics

---

## Tier 3: Research Judgment & Experimentation

This is often evaluated implicitly.

### 7. Experiment Design & Taste

You must demonstrate:

* Hypothesis-driven experiments
* Minimal, interpretable ablations
* Knowing when *not* to run something
* Learning from negative results

---

### 8. Reading & Applying Research

You are not expected to invent theory, but:

* Map papers to actionable training knobs
* Identify hidden assumptions
* Know when an idea will not scale or transfer

---

## Tier 4: Failure, Drift, and Emergent Behavior

This tier distinguishes senior researchers from strong engineers.

### 9. Objective–Behavior Coupling

* Why proxy objectives get gamed
* Reward shaping pathologies
* Detecting misalignment early
* Helpfulness / harmlessness / honesty tradeoffs

---

### 10. Continual & Iterative Training Risks

* Catastrophic forgetting vs slow drift
* Gradient interference across iterations
* Restart vs continue decisions
* Long-horizon evaluation strategies

---

### 11. Causal Reasoning in Training Changes

* Confounders in multi-knob experiments
* When A/B tests mislead in ML systems
* Isolating causal effects under noise
* “Correlation wins” vs “causal confidence”

---

### 12. Human-in-the-Loop Pathologies

* Annotator bias, fatigue, and incentives
* Preference noise and label drift
* Adaptive data collection feedback loops
* When to remove or redesign the human loop

---

### 13. Degradation, Drift, and Silent Failures

* Metric drift vs behavioral drift
* Offline vs online divergence
* Rare and long-tail failure modes
* Monitoring, rollback, and escalation criteria

---

## Tier 5: Architecture, Safety, and Strategic Judgment

This tier reflects **senior ownership**.

### 14. Architectural Limits & Inductive Bias

* What transformer inductive biases give you
* Long-context behavior limits
* When architecture (not tuning) is the bottleneck
* When architectural change is justified

---

### 15. Safety–Capability Tradeoffs

* Over-alignment and refusal calibration
* Capability suppression vs risk reduction
* Measuring “helpful but safe”
* When shipping a weaker model is correct

---

### 16. Compute Budgeting & Experiment Prioritization

* Scaling-law intuition for experiment sizing
* Downscaling vs skipping runs
* Iteration speed vs final quality
* Justifying compute spend

---

### 17. Owning Model Quality End-to-End

* Defining “good enough”
* Communicating uncertainty and risk
* Pushing back on shipping pressure
* Writing honest model limitation docs

---

## Tier 6: Pretraining Context (Targeted, Not Deep)

You are **not** a pretraining specialist, but you must understand:

### 18. Pretraining Choices That Surface Later

* Tokenization and data mixture effects
* Pretraining objective implications
* Why some post-training issues are “baked in”
* Limits of post-training compensation

---

## Tier 7: Execution & Proof of Skill (Critical Gap Closed)

This tier turns knowledge into **evidence**.

### 19. Project-Level Execution

You must be able to:

* Define a concrete model improvement goal
* Design evaluation *before* training
* Run controlled experiments
* Document failures and tradeoffs

---

### 20. Evaluation Harnesses & Tooling Literacy

You should understand:

* Common LLM eval harness patterns
* Prompt sensitivity and variance
* Leakage and benchmark overfitting
* When automatic evals fail

---

### 21. Training Runbooks & Debug Playbooks

You should have:

* “If loss diverges, do X”
* “If reward collapses, do Y”
* “If metrics disagree, do Z”
* Clear rollback criteria

---

### 22. Interview & Communication Readiness

You must be able to:

* Explain decisions crisply
* Walk through failures honestly
* Defend tradeoffs under pressure
* Reason aloud with incomplete data

---

## What You Explicitly Do NOT Need

Do not spend prep time on:

* Serving systems
* API design
* Frontend or UX
* Generic MLOps tooling
* Business KPI optimization
* Full-stack product ML

---

## End State

If you complete this syllabus, you are prepared to operate as:

> **A senior research engineer who can improve, debug, and own large models under real constraints — not just implement papers.**
