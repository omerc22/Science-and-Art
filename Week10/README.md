# Model Benchmarking: Flan-T5-Base Performance Report

![im](https://i.imgur.com/WMP2yc6.png)

This report documents the testing of the **Flan-T5-Base** (approx. 250M parameters) model across various chat and instruction-based scenarios. The goal was to observe how the model handles logic, persona, and summarization tasks under specific hyperparameter settings.

## 📊 Quick Summary of Findings

- **Reasoning:** Surprisingly decent logic, but fails at final arithmetic (calculated 10 + 3 = 11).
- **Persona:** Prone to hallucination. It creates fictional landmarks like "Staples Cathedral" in Istanbul.
- **Classification:** Excellent at few-shot sentiment analysis.
- **Summarization:** Tends to copy the first line of the input rather than distilling it.

---

## 🛠 Test Configurations & Results

### 1. General Chat & Greeting

* **Parameters:** `Temp: 0.7`, `Top-p: 0.9`
* **Observation:** The model maintained a friendly tone but got confused by the context, suddenly pivoting to a "hotel visit" response that wasn't in the prompt.

### 2. Logical Reasoning (Apples Math)

* **Prompt:** Chain of Thought (Step-by-step)
* **Parameters:** `Temp: 0.1`
* **Result:** 
    * *Model's Logic:* "5 + 2 = 10" (Incorrect) -> "10 + 3 = 11" (Incorrect).
    * *Takeaway:* Small base models struggle with internal math even when guided by step-by-step instructions.

### 3. Persona / Travel Guide

* **Prompt:** Act as a helpful travel guide for Istanbul.
* **Parameters:** `Temp: 0.5`
* **Result:** Initially answered "Ayn Rand International Airport" (Hallucination). On retry, it mentioned "Staples Cathedral" and "Sultanate of Istanbul".
* **Takeaway:** Base models lack factual grounding for specific travel advice and tend to invent names.

### 4. Few-Shot Sentiment Analysis

* **Prompt:** 3-example classification.
* **Parameters:** `Temp: 0.1`
* **Result:** `Positive`
* **Takeaway:** This is where the model shines. Pattern recognition in few-shot tasks is highly reliable.

### 5. Summarization Task

* **Prompt:** Customer Support Chat Summary.
* **Parameters:** `Temp: 0.3`
* **Result:** It simply repeated the customer's first sentence.
* **Takeaway:** For effective summarization, Flan-T5-Base requires a more forceful prompt or a lower `repetition_penalty`.

---

## Final Conclusion

Flan-T5-Base is highly capable for **classification** and **short-form extraction**, but it is not a "knowledge engine." For chat-based applications:

1. Keep the **Temperature low** (< 0.3) for factual tasks.
2. Use **Few-shot prompting** instead of Zero-shot whenever possible.
3. Expect creative "hallucinations" in creative writing tasks.

<br>

# Model Benchmarking: Stable Diffusion Art Generation Report

This document outlines the performance and stylistic behavior of my custom **Stable Diffusion Art Mode** model. The tests focus on two key parameters: **Guidance Scale** (how strictly the model follows the prompt) and **Inference Steps** (the amount of detail/refinement cycles).

## 📊 Evaluation Summary

- **Character Accuracy:** High recognition for famous figures (Keanu Reeves), though facial features distort at lower inference steps.
- **Contextual Blending:** The model successfully merges global figures with local textures (e.g., Keanu in Eminönü), though it struggles with the physics of small objects like bagels.
- **Architectural Interpretation:** Displays a heavy "Gothic/Classical" bias when prompted with historical locations.

---

#