# Generative Models

## Introduction
- Show pictures -> people understand what these models can do
- Establish what to expect -> - Prompt, architecture -> towards the end -> people understand the theme of this talk, and which topics come when (avoid question-spamming)

## Mathematical foundation
Question: Why we're starting with math? 
- The DDPM/variational view is confusing, and that's not how the SOTA models work -> people understand that diffusion doesn't mean DDPM anymore
- So, we'll start with the mathematical framework that makes it intuitive of what works
- And the framework also allows us to show how the different approaches to generative models connect with one another

Question: What is this framework?
- Monge formulation -> people understand the basic terminology of transport
- Kantorovich relaxation (need to motivate via examples what the coupling refers to) -> people understand what the "joint" achieves
- Wasserstein distance, IPM -> people understand how to measure distance, and how distance itself is formulated using "transport" ideas

Question: Wait, I know of generative models which didn't need transport?
  - Dual formulation (need to motivate via examples what the dual refer to) 
  - Briefly mention about how the dual formulation connects with WGAN, MMDGAN, mention my own paper -> people understand that I am a credible person to talk about this.

# The flow view:
Question: why are we talking about "flow"? I thought we're talking about transporting which can very well mean just teleporting?
- Benamou-Brenier theorem (motivate how they connect with the flow view) -> by this point, people understand why we're talking about velocity

## How do we generate samples?
- Establish position, velocity, ODE formulation, Euler -> by this point, people understand that if we had velocity, we can sample
- Mention how to tweak this to make this into the SDE formulation (which is diffusion), Euler-Nadaraya -> same

## How do we train models?
- ODE: simple, learning velocity directly
- Need to motivate the score-based view via diagrams. How do we get velocity from the score?
-> by this point people understand how to train these models, why SDE sampling is slow, and why SDE is still theoretically OT while the ODE suffers from discretisation and estimation errors.

## Now tell me how we "actually" train these models?
- Latent variable modeling
- VAE, neural architecture

## Guidance
- By this point, people know how prompting works, or seeding (image-to-image), or image-to-video works

## Closing remarks: consistency models, distillation, fine-tuning.
