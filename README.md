---

title: CIVERSE
emoji: 💻
colorFrom: purple
colorTo: blue
sdk: docker
app_file: server/app.py
pinned: false
tags: [openenv, rl, code-review, bug-detection, agent-eval]
-----------------------------------------------------------

# 💻 CIVERSE — Code Review RL Environment

### 🚀 Evaluating AI Agents on Real-World Code Review Tasks

[![OpenEnv](https://img.shields.io/badge/Powered_by-OpenEnv-brightgreen?style=for-the-badge)](#)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)](#)

---

## 🧠 Overview

**CIVERSE** is an OpenEnv-compatible reinforcement learning environment designed to evaluate how effectively AI agents perform **code review tasks**.

Instead of generating code, agents must:

* 🐞 Detect bugs
* 🏷️ Classify issues
* 🛠️ Suggest fixes

This transforms code review into a **structured RL problem**, enabling benchmarking of reasoning, precision, and correctness.

---

## ⚡ Core Idea

> This is NOT a bug detector.
> It is a **benchmark for evaluating AI code reviewers**.

Each episode presents a code snippet with hidden ground-truth bugs.
The agent interacts step-by-step and is scored based on accuracy and reasoning.

---

## 🎮 Action Space

Although internally abstracted, actions map directly to code review tasks:

| Action   | Interpretation        |
| -------- | --------------------- |
| `work`   | Detect a bug          |
| `focus`  | Classify the bug      |
| `switch` | Move to another issue |
| `break`  | No operation          |
| `delay`  | Skip step             |

### Example Action

```json
{"type": "work", "task_id": "m1"}
```

---

## 👁️ Observation Space

```json
{
  "code": "def add(a, b):\n    return a - b",
  "task_id": "e1",
  "step": 1
}
```

### What the agent sees:

* **code** → snippet to analyze
* **task_id** → scenario identifier
* **step** → current timestep

---

## 🧪 Task Levels

| Level     | Focus                              | Complexity              |
| --------- | ---------------------------------- | ----------------------- |
| 🟢 Easy   | Single bug detection               | Basic logic             |
| 🟡 Medium | Multiple bugs + classification     | Edge cases              |
| 🔴 Hard   | Detection + classification + fixes | Security + logic        |
| ⚫ Expert  | Multi-step reasoning               | Complex vulnerabilities |

---

## 🏆 Scoring System

```text
score = detection_accuracy × 0.33
      + classification_accuracy × 0.33
      + fix_quality × 0.34
```

### Metrics:

* **Detection Accuracy** → Did the agent find real bugs?
* **Classification Accuracy** → Did it label them correctly?
* **Fix Quality** → Are the fixes valid and meaningful?

---

## ⚙️ How It Works

```
reset() → Observation → Agent → Action → step() → Reward → repeat
```

* Environment provides code
* Agent responds with structured action
* System evaluates correctness
* Score is computed at episode end

---

## 🏗️ Project Structure

```
code-review-env/
├── models.py              # Core environment logic
├── inference.py           # LLM-based agent
├── openenv.yaml           # OpenEnv specification
├── Dockerfile             # Deployment config
├── backend/
│   └── main.py            # FastAPI server
├── server/
│   └── app.py             # Uvicorn entrypoint
├── grader/
│   └── code_review_graders.py
```

---

## 🚀 Running the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Start server

```bash
uvicorn server.app:app --port 7860 --reload
```

---

### 3️⃣ Run agent

```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

---

## 📊 Reward Design

| Event                  | Reward |
| ---------------------- | ------ |
| Correct detection      | +0.30  |
| Correct classification | +0.30  |
| Correct fix            | +0.40  |
| Incorrect action       | −0.10  |
| Skip                   | 0.00   |

---

## 🔧 Environment Variables

| Variable       | Description      |
| -------------- | ---------------- |
| `API_BASE_URL` | LLM endpoint     |
| `MODEL_NAME`   | Model identifier |
| `HF_TOKEN`     | API key          |

---

## 💡 Why This Matters

CIVERSE enables:

* 🧠 Evaluation of reasoning-heavy AI tasks
* 🔍 Benchmarking LLM code understanding
* ⚙️ Testing multi-step decision making

---

## 🏁 Final Note

This project demonstrates how **real-world developer workflows** can be converted into **reinforcement learning environments** — opening new directions for evaluating intelligent systems.

---
