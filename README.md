---
title: OpenEnv Code Review Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---



# CIVERSE(OPNE-ENV): Code Review RL Environment (OpenEnv-Compatible)

An **OpenEnv-compatible reinforcement learning environment** for evaluating AI agents on real-world code review tasks.

This project simulates a structured evaluation system where an AI agent must:

* 🐞 Detect bugs
* 🏷️ Classify bugs
* 🛠️ Suggest fixes

---

## 🧠 Project Overview

Unlike traditional code analysis tools, this system does **not detect bugs itself**.

Instead, it provides:

> ✅ A **benchmarking environment** that evaluates an agent’s ability to perform code review tasks.

Each code snippet comes with **ground-truth annotations**, and the agent is scored using **programmatic graders**.

---

## 🎯 Tasks

The environment consists of **3 progressive tasks**:

### 1. 🟢 Bug Detection (Easy)

* Determine whether code contains a bug
* Output: detect_bug or skip

### 2. 🟡 Bug Classification (Medium)

* Identify ALL bugs
* Classify:

  * Bug type (security, logic, etc.)
  * Severity (critical, high, medium, low)

### 3. 🔴 Fix Suggestion (Hard)

* Provide:

  * Correct fix
  * Explanation
* Evaluated using heuristic scoring

---

## ⚙️ Architecture

```
project/
│
├── environment.py        # RL environment (reset, step, state)
├── models.py            # Pydantic models (Action, Observation, Reward)
├── tasks.py             # Graders (evaluation logic)
├── snippet_generator.py # Dynamic code generation (OpenAI)
│
├── agent.py             # LLM-based agent (OpenAI)
├── baseline_inference.py# Full episode runner
│
├── api_routes.py        # Flask API endpoints
├── app.py               # Flask app entry point
```

---

## 🔁 RL Interaction Loop

```
reset() → Observation → Agent → Action → step() → Reward → repeat
```

---

## 📊 Evaluation System

### ✔ Task 1 & 2:

* Exact matching
* Precision / Recall / F1 scoring

### ✔ Task 3:

* Heuristic scoring:

  * Keyword overlap
  * Explanation quality
  * Code presence

---

## 🔍 Ground Truth

### Static Mode (No API Key)

* Uses **hardcoded code snippets**
* Each snippet contains predefined `known_bugs`

### Dynamic Mode (With OpenAI Key)

* Code + bugs are generated together using LLM
* Ground truth is **structured and validated**

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install flask flask-cors pydantic openai
```

---

### 2. Run the server

```bash
python app.py
```

Server runs at:

```
http://localhost:7860
```

---

## 🧪 API Testing (Postman)

### Reset environment

```
POST /api/reset
```

---

### Step (send action)

```
POST /api/step
```

Example:

```json
{
  "action_type": "detect_bug",
  "bug": {
    "line_number": 2,
    "bug_type": "security",
    "severity": "critical",
    "description": "SQL injection",
    "suggested_fix": "Use parameterized queries"
  },
  "confidence": 0.9
}
```

---

### Get state

```
GET /api/state
```

---

## 🤖 Running with Agent (Full Automation)

### Requires OpenAI API Key

```bash
export OPENAI_API_KEY="your_key"
python baseline_inference.py
```

---

### Output Example

```
Task 1 → Score: 1.0
Task 2 → Score: 0.76
Task 3 → Score: 0.65

Final Score: 0.79
```

---

## 🧠 Key Features

* ✅ OpenEnv-compatible (reset, step, state)
* ✅ Multi-step reasoning evaluation
* ✅ Structured action space
* ✅ Reward shaping (penalty + confidence bonus)
* ✅ Dynamic code generation
* ✅ Stateless + API-driven design

---

## ⚠️ Limitations

* Fix evaluation uses keyword matching (not semantic)
* Line-number-based matching can be brittle
* LLM-generated ground truth may not always be perfect

---

## 🚀 Future Improvements

* 🔥 Embedding-based semantic scoring (OpenAI / Ollama)
* 🔥 Multi-agent evaluation
* 🔥 Dataset persistence & benchmarking leaderboard
* 🔥 Visualization dashboard for agent performance

---

## 💡 Key Insight

> This project is not a bug detector — it is a **benchmark for evaluating AI code review agents**.

---

## 🏁 Hackathon Value

This system demonstrates:

* RL-style environment design
* LLM integration
* Structured evaluation
* Real-world applicability in code review automation

---

## 👨‍💻 Author

Built as part of a hackathon project focused on **AI evaluation systems and reinforcement learning environments**.

---

## ⭐ If you like this project

Give it a star and share feedback!
