<h1>
  <img src="static/favicon-square-dark.png#gh-light-mode-only" width="34" height="34" alt="consens.io">
  <img src="static/favicon-square.png#gh-dark-mode-only" width="34" height="34" alt="consens.io">
  consens.io
</h1>

<p align="center">
  <strong>Multi-model consensus for large language models.</strong>
</p>

<table align="center">
  <tr>
    <td align="center" width="90">
      <img src="static/icons/chatgpt.png" height="28" alt="OpenAI">
    </td>
    <td align="center" width="90">
      <img src="static/icons/claude.png" height="28" alt="Anthropic">
    </td>
    <td align="center" width="90">
      <img src="static/icons/gemini.png" height="28" alt="Gemini">
    </td>
    <td align="center" width="90">
      <img src="static/icons/mistral.png" height="28" alt="Mistral AI">
    </td>
    <td align="center" width="90">
      <img src="static/icons/deepseek.png" height="28" alt="DeepSeek">
    </td>
    <td align="center" width="90">
      <img src="static/icons/grok.png" height="28" alt="xAI Grok">
    </td>
    <td align="center" width="90">
      <img src="static/icons/glm.png" height="28" alt="GLM">
    </td>
    <td align="center" width="90">
      <img src="static/icons/kimi.png" height="28" alt="Kimi">
    </td>
  </tr>
</table>
**consens.io** is an experimental multi-model AI system for comparing and synthesizing responses from independent large language models.

Instead of relying on a single model, consens.io queries models from multiple providers independently and analyzes their responses to identify **agreement, disagreement, and uncertainty**. A separate synthesis step can then generate a consolidated answer from the collected outputs.

The project explores whether model diversity and aggregation can improve the robustness, transparency, and reliability of LLM-based answers.

## Method

1. A user query is evaluated independently by multiple language models.
2. Their responses are retained as separate observations.
3. Agreement and disagreement between the responses are analyzed.
4. A synthesis model produces a consolidated answer while preserving relevant differences.

consens.io currently integrates models from **OpenAI, Anthropic, Google, Mistral AI, DeepSeek, xAI, Zhipu AI / GLM, and Moonshot AI / Kimi**.

## Research

The repository includes experiments and benchmarks investigating multi-model aggregation strategies and the relationship between individual model performance and consensus-based answers.

Model agreement should not be interpreted as factual correctness. Consensus is treated as an additional signal rather than a substitute for source verification or empirical evidence.

## Stack

- Python / FastAPI
- JavaScript
- Firebase / Firestore
- Multiple LLM provider APIs

## Live

**[consens.io](https://consens.io)**

---

This repository contains the implementation of an actively developed experimental system. Models, prompts, aggregation methods, and evaluation procedures may change over time.
