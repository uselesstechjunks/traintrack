# Ideal Target Role

## Role Title (what you should search for)

Primary:

* Research Engineer (LLMs / Foundation Models)
* Research Scientist – Engineering Track

Secondary (selective):

* Applied Scientist (Model Quality / Training)
* Senior ML Engineer (Model Training / Systems for Training)

Avoid roles where the *primary* responsibility is downstream product integration.

---

## Core Mission of the Role

> **Design, train, evaluate, and improve large-scale models by directly operating on the training pipeline, objective functions, data, and optimization systems — with success measured by model-level metrics, not business KPIs.**

This is the single sentence your resume, job search, and prep plan should orbit.

---

## What You Actually Do (Day-to-Day)

You should expect most of your time to be spent on:

1. **Training-time interventions**

   * Pretraining, finetuning, continued training
   * Objective design (losses, regularization, auxiliary heads)
   * Curriculum, sampling, data mixing, filtering

2. **Model quality improvement**

   * Tradeoffs across metrics (accuracy, calibration, latency proxies, robustness, safety)
   * Ablations and controlled experiments
   * Debugging training failures and regressions

3. **Evaluation and diagnostics**

   * Offline eval design
   * Stress tests, slicing, error taxonomy
   * Interpreting metric movement causally

4. **Systems that exist *because* of research**

   * Training pipelines
   * Experiment orchestration
   * Scaling strategies
   * Memory / compute efficiency work where it affects model behavior

You are **not** primarily responsible for:

* Product feature delivery
* Serving APIs and SLAs
* A/B testing business metrics
* Feature engineering for downstream teams

---

## How Companies Will Evaluate You

They are implicitly asking:

1. Can you **improve a model that already works**?
2. Can you reason about **why a metric moved**?
3. Can you operate at **scale** without hand-holding?
4. Can you translate fuzzy goals (“better helpfulness”, “less hallucination”) into **concrete training changes**?

If a role does not evaluate you on these axes, it is not your role.

---

## Ideal Interview Loop (What You Should Expect)

Strong signal roles in this space usually test:

### 1. Model Understanding (High Weight)

* Training dynamics
* Loss functions and tradeoffs
* Failure modes of large models
* Scaling behavior

### 2. Experimentation & Debugging (High Weight)

* Given logs, metrics, or symptoms: what do you try next and why?
* How do you isolate causes?
* How do you avoid overfitting conclusions?

### 3. Systems-for-Training (Medium–High Weight)

* Distributed training intuition
* Memory vs compute tradeoffs
* Throughput vs convergence
* Fault tolerance in training

### 4. Coding (Medium Weight)

* Clean, numerically sane ML code
* Not LeetCode-heavy
* Often research-flavored implementation

### 5. Research Taste (Medium Weight)

* How you choose experiments
* What you would try if something fails
* How you decide what is “worth” exploring

---

## Resume Positioning (Non-Negotiable)

Your resume should read as:

> “Someone who improves models, not someone who ships features.”

Concrete guidance:

### Emphasize

* Training-scale work
* Objective changes
* Data or sampling decisions
* Offline metric improvements
* Experimental reasoning
* Systems choices made *because* of model behavior

### De-emphasize

* Business metrics
* Product impact language
* End-user features
* Generic MLOps tooling unless it enabled research

### Language Shift Example

Avoid:

> “Improved search relevance by X%”

Prefer:

> “Designed and evaluated ranking objectives under noisy feedback, improving offline NDCG stability and calibration under distribution shift”

---

## Job Description Filters (Fast Triage)

Apply **only if** at least 3–4 of the following are explicit:

* Mentions training or finetuning large models
* Mentions loss functions, objectives, or optimization
* Mentions evaluation beyond simple accuracy
* Mentions scaling, distributed training, or efficiency
* Mentions research experimentation
* Mentions collaboration with researchers

Reject immediately if:

* Role centers on deploying models
* Heavy emphasis on dashboards, KPIs, or A/B tests
* “End-to-end ownership of ML products” without training ownership
* Pure platform or infra support language

---

## “Things You Must Know” (Ruthlessly Scoped)

This is your **prep boundary**.

### A. Training Dynamics & Optimization

* SGD variants at scale
* Stability issues
* Curriculum and sampling effects
* Regularization in large models
* Why training fails and how to fix it

### B. Losses and Objectives

* Cross-entropy variants
* Ranking and contrastive losses
* Preference-based objectives
* Calibration-aware objectives
* Tradeoffs between competing metrics

### C. Evaluation & Debugging

* Offline eval design
* Metric sensitivity
* Data slicing
* Error taxonomies
* Avoiding false conclusions

### D. Systems for Training (Only What Affects Models)

* Data pipelines for training
* Distributed training basics
* Memory/computation tradeoffs
* Throughput vs convergence
* Reproducibility at scale

### E. Research Judgment

* How to choose experiments
* When not to run something
* How to interpret negative results
* How to generalize from small experiments

You do **not** need:

* Generic MLOps breadth
* Product analytics
* Extensive serving infra
* UI or feature pipelines

---

## Mental Model to Keep You Aligned

When deciding whether to study or apply to something, ask:

> “Does this make me better at improving a trained model under real constraints?”

If the answer is no, it is likely a distraction.

