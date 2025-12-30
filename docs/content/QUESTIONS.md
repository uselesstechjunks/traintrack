This is the right move. If you internalize the **right questions**, the reading becomes targeted, and interviews start to feel familiar rather than adversarial.

Below is a **question-first syllabus**, aligned exactly to:

* what strong Research Engineers are expected to *reason about* in interviews, and
* what you will actually face day to day when improving large models.

These are not exam questions. These are **operational thinking questions**.
If you can answer these crisply, you are at the right level.

You can paste these as headers or checklists next to each section in `READLIST.md`.

---

# 1. Post-Training & Preference Optimization (SFT, DPO, RLHF)

These questions are absolutely central.

## Foundational questions

* What *problem* is post-training solving that pretraining does not?
* When does SFT help, and when does it actively hurt a model?
* What assumptions does preference learning make about the data-generating process?
* What is the implicit objective being optimized in DPO?

## Deep understanding

* Why does DPO remove the need for an explicit reward model?
* In what regimes does DPO become unstable or misleading?
* How does preference noise propagate through training?
* What tradeoff is being made when optimizing helpfulness vs harmlessness vs honesty?

## Interview-style / real-world questions

* A model improves on preference metrics but regresses on factual accuracy. Why?
* How would you debug a post-training run that suddenly collapses?
* When would you choose GRPO over DPO?
* How do you know whether a gain is due to better alignment or overfitting to annotators?

---

# 2. Training Dynamics & Optimization at Scale

This is where “knows ML” becomes “can train models”.

## Foundational questions

* Why does increasing batch size change optimization behavior?
* What does learning rate actually control in large-scale training?
* What is gradient noise, intuitively?
* Why do large models often train “smoothly” until they suddenly don’t?

## Deep understanding

* How do batch size, learning rate, and data quality interact?
* Why does training sometimes diverge *late* rather than early?
* What does “stability” mean concretely in training runs?
* Why can two runs with identical configs diverge?

## Interview-style / real-world questions

* Loss is decreasing, but eval metrics are flat. What do you do?
* Throughput increased, but final model quality dropped. Why?
* How do you decide whether to stop a run early?
* What minimal changes would you try before restarting a multi-day job?

---

# 3. Evaluation & Model Debugging

This is often the **most heavily tested** area.

## Foundational questions

* Why is scalar evaluation fundamentally insufficient for LLMs?
* What does it mean for an evaluation to be “leaky”?
* What is the difference between measuring capability and measuring behavior?

## Deep understanding

* How do you design an evaluation that is hard to game?
* How do you decide whether an improvement is real or noise?
* What kinds of regressions are hardest to detect?
* How do preference-based evaluations fail?

## Interview-style / real-world questions

* Model A beats Model B on metric X. Would you ship it?
* A new model improves average score but worsens rare failure cases. What do you do?
* How would you debug a regression without retraining?
* What slices would you examine first, and why?

---

# 4. Data as a Training Lever

This is about *training behavior*, not pipelines.

## Foundational questions

* Why does data quality often matter more than model size?
* What does “data diversity” actually mean in practice?
* Why can adding more data make a model worse?

## Deep understanding

* How does sampling distribution affect learned behavior?
* When is filtering better than reweighting?
* How do curricula influence convergence and generalization?
* How does noisy data interact with preference optimization?

## Interview-style / real-world questions

* Performance regressed after adding new data. Why?
* How would you diagnose whether data or objective is the problem?
* Would you prefer fewer high-quality samples or more noisy ones?
* How do you prevent feedback loops in data collection?

---

# 5. Distributed Training & Systems Constraints

You are not an infra engineer, but you must reason under constraints.

## Foundational questions

* Why is model parallelism needed at all?
* What limits scaling more: memory or compute?
* Why do systems optimizations sometimes change model behavior?

## Deep understanding

* How do memory optimizations affect gradient quality?
* Why can faster training lead to worse convergence?
* What are the reproducibility pitfalls at scale?

## Interview-style / real-world questions

* A system optimization speeds up training but hurts final quality. Why?
* Training fails intermittently across nodes. What do you suspect?
* How would you design experiments under limited compute?
* What tradeoffs would you accept to get faster iteration?

---

# 6. Efficiency & Precision (Only Where It Matters)

Efficiency is a means, not a goal.

## Foundational questions

* Why does mixed precision usually work?
* When does reduced precision break training?
* What is the real cost of memory-saving tricks?

## Deep understanding

* How does numerical precision affect optimization dynamics?
* Why do some models tolerate aggressive quantization while others don’t?
* How do checkpointing strategies interact with training stability?

## Interview-style / real-world questions

* A mixed-precision run diverges but FP32 does not. Why?
* Would you trade precision for speed in early training?
* How do you decide what to optimize first: memory or throughput?

---

# 7. Experiment Design & Research Judgment

This is often evaluated implicitly.

## Foundational questions

* What makes an experiment “worth running”?
* How do you choose the *next* experiment?
* What does a good ablation look like?

## Deep understanding

* How do you avoid confirmation bias?
* How do you learn from negative results?
* When is it better to stop exploring a direction?

## Interview-style / real-world questions

* You have budget for 3 experiments. What do you run?
* An experiment shows mixed results. How do you interpret it?
* How do you know when you understand a failure?
* How do you communicate uncertainty to stakeholders?

---

# 8. Reading & Applying Research

This prevents wasted time.

## Foundational questions

* What part of this paper is actually actionable?
* What assumptions does this work rely on?
* What scale does this idea require?

## Deep understanding

* Why do many research ideas fail to transfer?
* How do you adapt a paper to your system?
* When should you ignore a paper entirely?

## Interview-style / real-world questions

* How would you test whether this paper’s idea applies here?
* What would you change before trying this at scale?
* Why might this approach fail in production?

---

# How to Use These Questions (Important)

For each reading section:

* Pick **3–5 questions**
* Try to answer them *before* reading
* Refine answers after reading
* Write answers as if explaining to a teammate

If you can do this comfortably, you are at **working-researcher level**, not “student” level.
