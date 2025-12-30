# Questions — Research Engineer / Research Scientist (Training-Focused Roles)

These questions define the **thinking standard** expected of a senior Research Engineer working on large-scale model training and improvement.

They are not trivia.
They reflect:

* real interview expectations, and
* real day-to-day decisions in post-training and model-quality teams.

If you can answer these crisply and defensibly, you are operating at the right level.

---

## Tier 0 — Foundations (Assumed, Not Tested Directly)

You should be able to answer these **without preparation**.

* How does backpropagation interact with stochastic optimization?
* Why do transformers scale better than many prior architectures?
* What are the basic failure modes of SGD-based training?
* What numerical issues arise in large neural networks?

If these are weak, fix them quietly and move on.

---

## Tier 1 — Core Model Improvement Skills

### 1. Post-Training & Preference Optimization (SFT, DPO, RLHF)

#### Foundational questions

* What problem does post-training solve that pretraining does not?
* When does SFT help, and when does it actively hurt a model?
* What assumptions does preference learning make about the data-generating process?
* What is the implicit objective being optimized in DPO?

#### Deep understanding

* Why does DPO remove the need for an explicit reward model?
* In what regimes does DPO become unstable or misleading?
* How does preference noise propagate through training?
* What tradeoff is being made when optimizing helpfulness vs harmlessness vs honesty?

#### Interview / real-world questions

* A model improves on preference metrics but regresses on factual accuracy. Why?
* How would you debug a post-training run that suddenly collapses?
* When would you choose GRPO over DPO?
* How do you distinguish genuine alignment gains from annotator overfitting?

---

### 2. Training Dynamics & Optimization at Scale

#### Foundational questions

* Why does increasing batch size change optimization behavior?
* What does learning rate actually control in large-scale training?
* What is gradient noise, intuitively?
* Why do large models often train smoothly until they suddenly do not?

#### Deep understanding

* How do batch size, learning rate, and data quality interact?
* Why does training sometimes diverge late rather than early?
* What does “stability” mean concretely in training runs?
* Why can two runs with identical configs diverge?

#### Interview / real-world questions

* Loss decreases but eval metrics are flat. What do you do next?
* Throughput improved but final model quality dropped. Why?
* How do you decide whether to stop a run early?
* What minimal changes would you try before restarting a multi-day job?

---

### 3. Evaluation & Model Debugging

#### Foundational questions

* Why is scalar evaluation insufficient for LLMs?
* What does it mean for an evaluation to be leaky?
* What is the difference between measuring capability and measuring behavior?

#### Deep understanding

* How do you design evaluations that are hard to game?
* How do you decide whether an improvement is signal or noise?
* What kinds of regressions are hardest to detect?
* How do preference-based evaluations fail?

#### Interview / real-world questions

* Model A beats Model B on metric X. Do you ship it?
* Average quality improves but rare failures worsen. What do you do?
* How would you debug a regression without retraining?
* What slices do you inspect first, and why?

---

### 4. Data as a Training Lever

#### Foundational questions

* Why does data quality often matter more than model size?
* What does data diversity actually mean in practice?
* Why can adding more data make a model worse?

#### Deep understanding

* How does sampling distribution affect learned behavior?
* When is filtering better than reweighting?
* How do curricula influence convergence and generalization?
* How does noisy data interact with preference optimization?

#### Interview / real-world questions

* Performance regressed after adding new data. Why?
* How do you tell whether data or objective is the problem?
* Would you prefer fewer clean samples or more noisy ones?
* How do you prevent feedback loops in data collection?

---

## Tier 2 — Systems for Training (Only What Matters)

### 5. Distributed Training & Constraints

#### Foundational questions

* Why is model parallelism needed at all?
* What limits scaling more: memory or compute?
* Why do system optimizations sometimes change model behavior?

#### Deep understanding

* How do memory optimizations affect gradient quality?
* Why can faster training lead to worse convergence?
* What are reproducibility pitfalls at scale?

#### Interview / real-world questions

* A system optimization speeds up training but hurts quality. Why?
* Training fails intermittently across nodes. What do you suspect?
* How do you design experiments under limited compute?
* What tradeoffs do you accept to get faster iteration?

---

### 6. Efficiency & Numerical Precision

#### Foundational questions

* Why does mixed precision usually work?
* When does reduced precision break training?
* What is the real cost of memory-saving tricks?

#### Deep understanding

* How does numerical precision affect optimization dynamics?
* Why do some models tolerate aggressive quantization?
* How do checkpointing strategies interact with stability?

#### Interview / real-world questions

* Mixed precision diverges but FP32 does not. Why?
* Would you trade precision for speed early in training?
* How do you decide what to optimize first: memory or throughput?

---

## Tier 3 — Research Judgment

### 7. Experiment Design & Taste

#### Foundational questions

* What makes an experiment worth running?
* How do you choose the next experiment?
* What does a good ablation look like?

#### Deep understanding

* How do you avoid confirmation bias?
* How do you learn from negative results?
* When is it better to stop exploring a direction?

#### Interview / real-world questions

* You have budget for three experiments. What do you run?
* Results are mixed. How do you interpret them?
* How do you know when you understand a failure?
* How do you communicate uncertainty?

---

### 8. Reading & Applying Research

#### Foundational questions

* What part of this paper is actionable?
* What assumptions does it rely on?
* What scale does this idea require?

#### Deep understanding

* Why do many ideas fail to transfer?
* How do you adapt a paper to your system?
* When should you ignore a paper?

#### Interview / real-world questions

* How would you test whether this idea applies here?
* What would you change before scaling it?
* Why might this fail in practice?

---

## Tier 4 — Failure, Drift, and Emergent Behavior

### 9. Objective–Behavior Coupling

* Why do proxy objectives get gamed?
* Why do small loss changes cause large behavior shifts?
* How do objectives interact with distribution shift?
* When do you accept imperfect alignment?

---

### 10. Continual & Iterative Training

* Why does repeated fine-tuning cause forgetting?
* How does gradient interference accumulate?
* When do you restart vs continue training?
* How do you reason about training debt?

---

### 11. Causal Reasoning in Model Improvements

* Why are most improvements correlational?
* What confounders dominate large-scale experiments?
* When does ablation mislead?
* What causal confidence is enough to ship?

---

### 12. Human-in-the-Loop Pathologies

* Why does preference data decay?
* How do annotator incentives distort learning?
* When should humans be removed from the loop?
* How do you redesign annotation pipelines?

---

### 13. Degradation, Drift, and Silent Failures

* Why do models degrade without code changes?
* Metric drift vs behavioral drift
* Offline vs online divergence
* Rollback vs iterate decisions

---

## Tier 5 — Architecture, Safety, and Strategic Judgment

### 14. Architectural Inductive Biases

* What transformer biases give you and what they limit
* When architecture dominates over tuning
* How to detect architectural bottlenecks

---

### 15. Safety–Capability Tradeoffs

* Why alignment suppresses capability
* Over-alignment and refusal calibration
* When shipping a weaker model is correct

---

### 16. Compute Budgeting & Prioritization

* How scaling laws inform experiment sizing
* One big run vs many small runs
* When to kill a partially successful run

---

### 17. Owning Model Quality End-to-End

* What “good enough” means
* Communicating risk honestly
* Pushing back on premature shipping
* Writing limitation docs

---

## Tier 6 — Pretraining Context (Targeted)

### 18. Pretraining Choices That Surface Later

* Tokenization and data mixture consequences
* Objective choices baked into representations
* What post-training cannot fix

---

## Tier 7 — Execution & Proof of Skill

### 19. Project Execution

* How to define success before training
* How to design evals first
* How to document failures

---

### 20. Evaluation Harness Literacy

* Common eval harness failure modes
* Prompt sensitivity and variance
* Benchmark overfitting

---

### 21. Training Runbooks & Debug Playbooks

* If loss diverges, what do you do?
* If reward collapses, what do you do?
* If metrics disagree, what do you trust?

---

### 22. Interview & Communication Readiness

* Explaining tradeoffs crisply
* Defending decisions under pressure
* Reasoning aloud with incomplete data

---

## End State

If you can answer these questions convincingly, you are prepared to function as:

> **A senior research engineer who owns model quality, not just model code.**