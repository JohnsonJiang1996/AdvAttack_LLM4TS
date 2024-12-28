# 🔒 Adversarial Attacks on Time Series LLMs 🚀

Welcome to the **Adversarial Time Series LLMs** repository! 
LLM-TimeSeriesVulnerabilities is a research project focused on investigating the adversarial vulnerabilities of Large Language Models (LLMs) in time series forecasting tasks. This repository introduces a targeted adversarial attack framework tailored to LLM-based time series models, specifically utilizing gradient-free optimization techniques to create minimal yet highly effective perturbations. The attacks significantly degrade the accuracy of time series predictions across multiple LLM architectures, including GPT-3.5, GPT-4, LLaMa, and Mistral.
---

## 📚 What’s This All About?

LLMs have made waves in time series forecasting—handling everything from predicting stock trends to weather patterns with their uncanny ability to process sequential data. But here's the catch: **they aren't as invincible as they seem**.

This repo dives deep into:

- 🚧 How adversarial attacks can break LLMs’ predictive prowess.
- 🛠️ A targeted adversarial attack framework for LLM-based forecasting models.
- 📉 Experiments demonstrating how subtle data perturbations can turn robust predictions into a chaotic mess of randomness.

---

## ✨ Key Features

- **Black-Box Magic**: Crafting adversarial attacks without peeking inside the LLM’s inner workings.
- **Directional Gradient Approximation (DGA)**: Our secret sauce to transform time series forecasts into random walks.
- **Benchmark Datasets**: Tested across ETTh1, IstanbulTraffic, and more—our attacks don’t discriminate!
- **Model Variety**: From fine-tuned LLaMa and GPT-4 to specialized TimeGPT, no LLM is safe!

---

## 🎯 The Core Idea

Adversarial attacks introduce tiny tweaks to time series input data—imperceptible to the human eye but devastating to LLMs’ predictions. Imagine stock prices turning into white noise or weather forecasts spiraling into gibberish. That’s the power of **DGA**. 

> 📊 Results? Adversarial perturbations disrupt predictions much more than regular Gaussian noise, revealing critical vulnerabilities in LLMs.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/your-username/adversarial-ts-llms.git
cd adversarial-ts-llms
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

## 🧪 Experiments Galore

| Dataset            | Models Tested        | Attack Type | Impact (MAE 🁑) |
|--------------------|----------------------|-------------|----------------|
| **ETTh1**          | GPT-3.5, GPT-4      | DGA         | Significant    |
| **IstanbulTraffic**| LLaMa, Mistral      | DGA         | Very High      |
| **Weather**        | TimeGPT, LLM-Time   | DGA         | Substantial    |

> For detailed results and visualizations, check out our [experiments folder](experiments/).

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

---

## 📜 Citation

If you use this code or find the research helpful, please cite:
```
@article{jiang2024adversarial,
  title={Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting},
  author={Jiang, Sicong and Liu, Fuqiang, et al.},
  journal={Preprint},
  year={2024}
}
```

---

🔍 Explore. Experiment. Attack. Let’s make forecasting stronger, one adversarial example at a time!
