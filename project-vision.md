# AI Model Hub — Project Vision & Technical Reference

> This document exists to preserve the full vision, technical decisions, and research ideas behind this project. If this is being read in a new AI conversation, use it to continue guiding the developer (Jdlynchv) from wherever they left off.

---

## The Developer

- GitHub username: `jdlynchv`
- Repository: `ai-model-hub`
- Background: Developer and product-minded builder, beginner-to-intermediate level, learning as the project is built
- Goal: Build a valuable, open source platform with genuine research and acquisition potential, while developing deep ML infrastructure skills

---

## Project Overview

An open source, free web platform where anyone can create a fully customized AI language model built on top of existing open source base models (Llama, Mistral, Phi, Qwen, etc.). The platform has two modes: a guided/assisted mode for casual non-technical users, and a full developer mode with deep configuration control.

The platform is not just a fine-tuning tool. Its deeper purpose is to systematically collect structured metadata about every training run — what configurations were used, what data was trained on, what benchmark results were achieved — and accumulate this into a research-grade dataset that maps "configuration → performance" relationships across domains and model types. This dataset is the core long-term asset of the project.

---

## Core Features

### 1. Model Builder
Users select a base open source model, upload or reference a dataset, configure fine-tuning parameters through a UI, and launch a training job. Casual users get a simplified guided flow with explanations. Developers get full parameter control including LoRA rank, alpha, dropout, target modules, learning rate, batch size, and more.

### 2. Public Model Hub
Every trained model gets a public profile page showing its full configuration, training data info, benchmark scores, and a demo interface to try it. Models can be downloaded, shared, or forked by other users.

### 3. Automatic Benchmarking
Every model is automatically evaluated against standardized benchmarks relevant to its domain upon completion. Math models run on GSM8K, code models on HumanEval, general models on MMLU, etc. Importantly, every model is also run against a broad set of tasks outside its stated domain to capture unexpected capability changes.

### 4. Leaderboards and Competition
Models are ranked by benchmark performance within each domain. This drives engagement and surfaces the best-performing configurations publicly.

### 5. Metadata Database (Core Asset)
Every training run logs a structured record including all configuration details, dataset characteristics, benchmark scores across all domains, and training outcome. This database grows with every user interaction and is the primary research and commercial value of the platform.

---

## Research Ideas (To Be Built After Core Platform)

These three ideas are planned extensions that emerge naturally from the platform's data. The core platform must be built first, but certain data collection decisions must be made from day one to enable them later.

### Idea 1: Failure Museum
A systematic, public record of failed or poor-performing training runs with their full configuration. Nobody captures failure data at scale. This is scientifically valuable because it maps the boundaries of what works — researchers spend enormous resources rediscovering failure modes. Build this once the platform has scale and failed run data has accumulated.

**Data needed from day one:** training loss curves, early stopping events, error states, user-reported success/failure flag (simple thumbs up/down after seeing benchmark results), and preservation of low-scoring runs rather than filtering them out.

### Idea 2: Dataset Genealogy Tracking
A graph that tracks the lineage of every model on the platform — which base model it started from, which dataset it was trained on, whether it was forked from another user's model, and how capabilities changed at each step. This creates a traceable map of how training data shapes model behavior over time, which is valuable for AI safety, interpretability research, and understanding capability transfer.

**Data needed from day one:** a `parent_model_id` field in the models database table (null if trained from a base model, references another model's ID if forked), and a strict forking system that always records parent-child relationships explicitly.

### Idea 3: Emergent Capability Detection
When a model is fine-tuned for one task, it sometimes gets unexpectedly better or worse at unrelated tasks. Because every model on the platform is benchmarked across multiple domains automatically, the platform is in a unique position to detect these patterns at scale. A layer that surfaces and flags unexpected capability changes — "this math model's translation score improved 15%" — turns the platform into a research observatory. Findings can be published and cited.

**Data needed from day one:** benchmark scores stored for ALL domains for every model, not just the relevant ones. Store them in a flexible JSON field so no domain data is ever discarded.

---

## Critical Database Fields (Must Be Included From Day One)

These four fields in the models/training runs table enable all three research ideas above:

- `parent_model_id` — null if from a base model, foreign key to another model if forked
- `training_outcome` — enum: success / failure / partial
- `loss_curve_data` — stored as JSON, full curve not just final loss
- `benchmark_scores` — flexible JSON object storing scores for ALL domains tested, not just the primary one

---

## Technical Stack

### Frontend
- **React** with **Next.js** — UI framework and routing
- **Tailwind CSS** — styling
- **shadcn/ui** — pre-built component library
- **Hosting:** Vercel (free tier to start)

### Backend
- **Python** with **FastAPI** — server logic and API
- Handles: user auth, dataset uploads, triggering training jobs, storing metadata
- **Hosting:** Railway or Render

### Database
- **Supabase** — PostgreSQL database, authentication, and file storage in one
- Stores: user accounts, model metadata, benchmark results, dataset references, all training run records

### Compute (GPU for fine-tuning)
- **Modal.com** — run GPU workloads on demand in Python, pay only when jobs run
- Later alternatives: RunPod, Lambda Labs for cheaper raw GPU rental at scale

### AI/ML Libraries
- **Hugging Face Transformers** — loading and running open source base models
- **Hugging Face PEFT** — LoRA and other parameter-efficient fine-tuning methods
- **Hugging Face Datasets** — loading and processing training data
- **PyTorch** — underlying ML framework everything is built on

### Benchmarking
- **EleutherAI lm-evaluation-harness** — standard open source evaluation framework, runs models against hundreds of standardized tasks

---

## Fine-Tuning Approach

The platform uses **LoRA (Low-Rank Adaptation)** as the primary fine-tuning method. Instead of retraining all parameters in a large model (extremely expensive), LoRA adds small adapter layers and trains only those. This makes fine-tuning accessible without massive compute budgets.

Key LoRA parameters users will configure: rank (r), alpha, dropout, target modules. The platform should explain these in plain language for casual users and expose them fully for developer users.

**Supported base models (initial):** Llama 3, Mistral, Phi-3, Qwen

---

## Development Phases

**Phase 0 — Environment Setup** ✅ COMPLETE
VS Code, Python 3.14.2, pip 25.3, Git 2.49.0, GitHub repo created (ai-model-hub), virtual environment set up inside project.

**Phase 1 — Python and ML Fundamentals**
Learn enough Python to be productive. Work through the Hugging Face course at huggingface.co/learn. Get comfortable loading models, running inference, and doing basic fine-tuning in Google Colab notebooks.

**Phase 2 — Backend Infrastructure**
Set up Supabase with the correct schema (including the four critical fields above). Build a basic FastAPI backend. Implement user authentication. Design and create the metadata schema from the start.

**Phase 3 — Fine-Tuning Pipeline**
Integrate Modal for GPU compute. Build the pipeline that takes a dataset and parameters, runs a LoRA fine-tuning job on a chosen base model, and logs all metadata on completion.

**Phase 4 — Benchmarking System**
Integrate lm-evaluation-harness. Automatically run every new model through relevant benchmarks plus a broad set of cross-domain benchmarks. Store all results in the metadata schema.

**Phase 5 — Frontend**
Build the React/Next.js interface: model builder UI, hub, leaderboards, model profile pages, demo interface. Build developer mode first, then layer on the guided casual user mode.

**Phase 6 — Polish and Launch**
Forking system with lineage tracking, model sharing, public launch. Seed the platform with example models across different domains to avoid cold start problem.

**Phase 7 — Research Layer**
Once data is accumulating: build Emergent Capability Detection (Idea 3) first, then Failure Museum (Idea 1), then Dataset Genealogy (Idea 2). Publish findings. This is where the platform transitions from tool to research observatory.

---

## Competitive Landscape

- **Hugging Face** — largest overlap, model hub and sharing, but no systematic metadata collection or guided creation. Well resourced, biggest competitive risk.
- **Weights & Biases** — experiment tracking and metadata, no community hub or benchmarking competition element
- **Predibase / Lamini** — fine-tuning as a service, enterprise focused, not community oriented
- **Together.ai** — fine-tuning on open source models, no hub or research angle

**Differentiators:** guided UX for non-developers, systematic metadata collection from day one, automatic cross-domain benchmarking, research ideas (failure museum, genealogy, emergent capabilities) that no one else is building openly.

---

## Commercial Value Thesis

The platform's value is not primarily the tool itself — it is the structured dataset of configuration → performance mappings accumulated over thousands of training runs. This dataset:

- Does not exist publicly in clean structured form anywhere
- Is useful to AI researchers studying fine-tuning dynamics
- Is useful to AI labs understanding capability transfer
- Becomes more valuable the larger it grows
- Can be licensed, published as research, or used as acquisition leverage

Realistic near-term outcomes: respected open source project with active community, dataset licensing to researchers, academic publication partnerships. Longer-term: acquisition interest from AI labs or research organizations once operating at significant scale.

---

## Key Learning Resources

- Python: "Python for Everybody" on Coursera (free to audit)
- ML concepts: 3Blue1Brown "Neural Networks" series on YouTube
- Transformers intuition: Andrej Karpathy "Let's build GPT from scratch" on YouTube
- Hugging Face ecosystem: huggingface.co/learn (free, covers all core libraries)
- GPU experimentation: Google Colab (free GPU access for notebooks)

---

*Last updated: February 2026. Continue guiding the developer through the phases in order, reminding them to implement the four critical database fields and cross-domain benchmarking when reaching Phase 2 and Phase 4 respectively.*
