# MIR-Bench
# MIR-Bench: A Chinese Benchmark for Converting Natural Language to Executable Medical Insurance Rules

<div align="center">

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/YOUR_ARXIV_ID_HERE)

</div>

## 📖 Introduction

[cite_start]**MIR-Bench** is the first comprehensive Chinese benchmark designed to evaluate the capability of Large Language Models (LLMs) in converting natural language **Medical Insurance Rules (MIR)** into executable rule functions[cite: 50].

Effective medical insurance supervision relies on the precise execution of complex rules. [cite_start]While LLMs demonstrate significant potential in general NLP tasks, they often struggle with the domain-specific rigor required for medical governance[cite: 49]. MIR-Bench bridges this gap by providing a robust evaluation platform consisting of:

* [cite_start]**2,115** real-world medical insurance rules derived from policy documents[cite: 51].
* [cite_start]**5,636** execution-based patient test cases[cite: 51].
* [cite_start]A **Dual Evaluation Framework** assessing both **AST structure correctness** (Rule Level) and **execution consistency** (Execution Level)[cite: 52].

### Task Overview
[cite_start]The benchmark covers three major categories of violations [cite: 372-375]:

1.  **Excessive Charges (超标准收费)**: Price calculation and unit constraints.
2.  **Duplicate Charges (重复收费)**: Mutually exclusive billing items (parent-child relationships).
3.  **Ineligible Drug Use (用药限制)**: Restrictions based on age, gender, diagnosis, etc.

![Benchmark Pipeline](assets/pipeline_figure.png)

## 📂 Repository Structure
