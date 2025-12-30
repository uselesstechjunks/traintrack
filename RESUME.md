
# Resume + Project Strategy

## Resume: How It Must Read

Your resume should scream:

> “I improve models, not features.”

### Structural Changes (Important)

**Top section**

* Short 2–3 line summary explicitly mentioning:

  * Model training
  * Evaluation
  * Post-training / optimization
  * Large-scale systems (only in training context)

**Experience bullets**
Every bullet should answer:

* What model?
* What lever did you pull?
* What metric moved?
* Why did it work?

---

### Language Transformations (Examples)

Avoid:

> “Built ML pipelines for X product used by Y users”

Prefer:

> “Designed and evaluated training objectives under noisy feedback, improving offline calibration and stability across slices”

Avoid:

> “Improved CTR by 4%”

Prefer:

> “Diagnosed metric regression via data slicing and loss analysis; introduced curriculum-based sampling to stabilize convergence”

---

## Projects: What Actually Helps Shortlisting

You do **not** need many projects.
You need **1–2 very pointed ones**.

### Project Type 1: Post-Training Playground (High Value)

* Small to medium LLM
* Implement SFT + DPO
* Compare objectives
* Analyze failure modes
* Show metric tradeoffs

Focus on:

* Evaluation methodology
* Reasoning behind decisions
* Not chasing SOTA

---

### Project Type 2: Training Diagnostics Tool (Optional but Strong)

* Tool to analyze training logs
* Detect instability, overfitting, or regressions
* Even a small prototype is enough

This signals:

* Research judgment
* Systems awareness
* Debugging ability

---

## What Will NOT Help Much

* Generic Kaggle projects
* Full-stack demos
* Toy chatbots
* Inference-only optimizations
* Model deployment dashboards

---

## Final Alignment Check (Use This Weekly)

Ask yourself:

> “If I were asked to improve a trained LLM tomorrow, would this prep help me?”

If the answer is no — drop it.