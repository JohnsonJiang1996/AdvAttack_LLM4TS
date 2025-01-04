# 🔒 Adversarial Attack on LLM4TS🚀

Welcome to the **Adversarial Attack on LLM4TS** repository! This repository contains the official code implementation for the paper [Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting](https://arxiv.org/abs/2412.08099). Dive into the fascinating world of Large Language Models (LLMs) and their application in time series forecasting as we explore their capabilities, limitations, and the vulnerabilities exposed by adversarial attacks.

---

## 📜 Citation

If you use this code or find the research helpful, please cite:
```
@article{liu2024adversarial,
  title={Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting},
  author={Liu, Fuqiang and Jiang, Sicong and Miranda-Moreno, Luis and Choi, Seongjin and Sun, Lijun},
  journal={arXiv preprint arXiv:2412.08099},
  year={2024}
}
```

---

## 📚 What’s This All About?

LLMs have made waves in time series forecasting—handling everything from predicting stock trends to weather patterns with their uncanny ability to process sequential data. But here's the catch: **they aren't as invincible as they seem**.

This repo dives deep into:

- 🚧 How adversarial attacks can break LLMs’ predictive prowess.
- 🛠️ A targeted adversarial attack framework for LLM-based forecasting models.
- 📉 Experiments demonstrating how subtle data perturbations can turn robust predictions into a chaotic mess of randomness.

---

## ✨ Key Features

- **Black-Box Attack**: Crafting adversarial attacks without peeking inside the LLM’s inner workings.
- **Directional Gradient Approximation (DGA)**: Our proposed attack methods for those LLM4TS models.
- **Benchmark Datasets**: Tested across ETTh1, IstanbulTraffic, and more—our attacks don’t discriminate!
- **Model Variety**: From fine-tuned LLaMa and GPT-4 to specialized TimeGPT, no LLM is safe!

---

## 🎯 The Core Idea

Adversarial attacks introduce tiny tweaks to time series input data—imperceptible to the human eye but devastating to LLMs’ predictions. Imagine stock prices turning into white noise or weather forecasts spiraling into gibberish. That’s the power of **DGA**. 


---

## 🚀 Getting Started

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/your-username/AdvAttack_LLM4TS.git
cd AdvAttack_LLM4TS
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Try It Out
Run our experiments on your favorite LLM and dataset combo:
```bash
python run_attack.py --model LLMTime --dataset ETTh1 --attack DGA
```

---

## 🧪 Experiment Summary Table

| Dataset            | Models Tested                                           | Attack Type | Impact (MAE & MSE)                      |
|--------------------|---------------------------------------------------------|-------------|-----------------------------------------|
| **ETTh1**          | GPT-3.5, GPT-4, LLaMa 2, Mistral, GPT-2 (Time-LLM)     | DGA         | Consistently increased MAE & MSE  |
| **ETTh2**          | GPT-3.5, GPT-4, LLaMa 2, Mistral, GPT-2 (Time-LLM)     | DGA         | High impact on LLMs, especially Mistral |
| **IstanbulTraffic**| GPT-3.5, GPT-4, LLaMa 2, Mistral, GPT-2 (Time-LLM)     | DGA         | Very high impact, particularly on Mistral|
| **Weather**        | GPT-3.5, GPT-4, LLaMa 2, Mistral, GPT-2 (Time-LLM)     | DGA         | Minimal effect; DGA impact consistent   |
| **Exchange**       | GPT-3.5, GPT-4, LLaMa 2, Mistral, GPT-2 (Time-LLM)     | DGA         | Moderate impact; GPT-4 slightly better  |

> Detailed results available in the [experiments folder](experiments/).


---

## 📊 Highlights

- **Figure 2**: Visual proof of chaos—see how predictions deviate under attack!
- **Table 1**: Side-by-side performance comparisons under clean vs. adversarial inputs.
- **Hyperparameter Insights**: Fine-tune perturbation scale for maximum disruption.

---

## 🛡️ Why This Matters

As we step closer to LLMs ruling the world (or at least our forecasts), **robustness matters**. This research sheds light on the vulnerabilities of LLMs in time-sensitive domains and paves the way for building defenses against malicious adversarial attacks.

---

## ❤️ Contributing

Got ideas for stronger attacks? Know how to fortify LLMs? We’d love your help! Fork the repo, submit PRs, or just drop by with your feedback.

---

## 👩‍🔬 Authors

- **Fuqiang Liu**, **Sicong Jiang** (Co-First Authors)
- **Seongjin Choi**, **Luis Miranda-Moreno**, **Lijun Sun**


🔍 Explore. Experiment. Attack. Let’s make forecasting stronger, one adversarial example at a time!
