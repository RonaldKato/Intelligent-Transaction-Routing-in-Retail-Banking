# Intelligent Transaction Routing in Retail Banking

<img width="100%" alt="Intelligent Transaction Routing Pipeline" src="https://github.com/user-attachments/assets/3d5c1ca7-f489-43a9-b2b7-124eab72d252" />

A reproducible, simulation-based framework for evaluating transaction-routing architectures in retail banking through synthetic, industry-parameterized transaction workloads. The project compares traditional Transaction Switch routing, API Gateway routing, and Machine-Learning Hybrid (ML-Hybrid) routing under controlled experimental conditions.

## Overview

Retail banking systems increasingly rely on API-based architectures to support scalability, interoperability, and efficient transaction processing. However, the performance implications of alternative routing architectures and the additional value of machine-learning-assisted routing require systematic evaluation.

This project investigates:

* Whether API Gateway routing provides measurable performance advantages over traditional Transaction Switch architectures.
* Whether machine-learning-based routing can further optimize transaction processing when both routing options are available.
* How architectural performance varies across transaction types and banking channels.
* Whether observed performance differences are statistically significant and practically meaningful.

The experimental framework simulates **60,000 synthetic banking transactions** across five retail banking channels:

| Banking Channel                |
| ------------------------------ |
| Point of Sale (POS)            |
| Automated Teller Machine (ATM) |
| E-commerce                     |
| Mobile Application             |
| USSD                           |

The simulation includes six transaction types:

| Transaction Type |
| ---------------- |
| PURCHASE         |
| WITHDRAWAL       |
| BALANCE_INQUIRY  |
| FUND_TRANSFER    |
| BILL_PAYMENT     |
| REVERSAL         |

## Routing Architectures

The framework evaluates three routing strategies.

| Strategy        | Description                                                                              |
| --------------- | ---------------------------------------------------------------------------------------- |
| **Switch-only** | Routes all transactions through the traditional Transaction Switch.                      |
| **API-only**    | Routes all transactions through the API Gateway.                                         |
| **ML-Hybrid**   | Uses a machine-learning router to select between the Transaction Switch and API Gateway. |

The ML-Hybrid architecture is evaluated to determine whether intelligent routing provides measurable improvements over the individual routing alternatives.

## Evaluation Metrics

The experimental evaluation considers performance, reliability, routing behavior, and statistical validity.

### Performance and Reliability

* Transaction latency
* Throughput capacity
* Transaction success rate
* Workload sensitivity
* Routing distribution and decisions

### Statistical Analysis

* Paired *t*-tests
* Wilcoxon signed-rank tests
* Cohen's *d* effect size
* Bootstrap confidence intervals
* Statistical significance testing

### Machine Learning Evaluation

* Classification performance
* Routing decision analysis
* Feature importance
* Comparative performance against API-only routing

## Methodology

The project implements a coordinated Python experimentation pipeline that:

1. Generates and validates a synthetic banking transaction workload.
2. Engineers features for transaction routing.
3. Simulates the Switch-only, API-only, and ML-Hybrid architectures under controlled conditions.
4. Trains and tunes the machine-learning routing model.
5. Evaluates architectural performance using statistical and machine-learning metrics.
6. Produces experimental outputs to support reproducibility and research analysis.

Paired statistical procedures account for the paired transaction structure when comparing architectural performance. The analysis combines performance benchmarking, hypothesis testing, effect-size estimation, and bootstrap-based uncertainty analysis.

## Key Findings

Under the modeled simulation conditions, the study reports the following results:

| Metric                                  |                   Result |
| --------------------------------------- | -----------------------: |
| API Gateway latency                     |                199.60 ms |
| Transaction Switch latency              |                255.35 ms |
| Reported latency reduction              |                   21.83% |
| API throughput proxy                    |     226.5 transactions/s |
| Switch throughput proxy                 |     177.0 transactions/s |
| Reported throughput improvement         |      Approximately 28.0% |
| Transaction success rates               | Statistically comparable |
| ML-Hybrid improvement over API-only     |      Approximately 0.15% |
| Transactions routed to API by ML-Hybrid |      Approximately 99.5% |

### Interpretation

Under the modeled conditions, the results suggest that **architectural modernization contributes more substantially to performance improvement than machine-learning-based routing when the API Gateway already outperforms the traditional Transaction Switch**.

The limited additional improvement observed with ML-Hybrid routing highlights the importance of evaluating the incremental value of intelligent routing against its implementation complexity and operational requirements.

> **Important:** These findings are specific to the simulation parameters and synthetic workload used in this study. They should not be interpreted as direct measurements of production banking infrastructure.

## Project Structure

The primary experimental entry point is:

```text
run_all.py
```

The complete pipeline should be executed through this entry point to ensure that the experimental workflow is coordinated consistently.

> Ensure that `run_all.py` is present in the repository and that all execution instructions correspond to the current implementation.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/RonaldKato/Intelligent-Transaction-Routing-in-Retail-Banking.git
cd Intelligent-Transaction-Routing-in-Retail-Banking
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

Install the dependencies specified in the repository's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Run the Experimental Pipeline

Execute the complete experimental workflow using:

```bash
python run_all.py
```

The pipeline should generate the relevant experimental outputs and statistical analyses as implemented in the repository.

## Reproducibility

The experiments use parameterized synthetic transaction data and controlled simulation conditions to support reproducible evaluation.

Reproducibility considerations include:

* Synthetic workload generation.
* Controlled architectural simulation parameters.
* Consistent evaluation procedures.
* Documented Python dependencies.
* A unified experimental execution entry point.

The dataset does not represent production banking traffic. Results are dependent on the assumptions, parameter values, and simulation design adopted in the study.

## Ethical and Data Protection Considerations

This project uses fully synthetic transaction data.

* No customer information is included.
* No personally identifiable information (PII) is processed.
* No production banking records are used.
* No confidential institutional data is included.

The framework is intended to support responsible research and experimentation without exposing sensitive banking or customer information.

## Research Context

This repository supports research in:

* Retail banking transaction infrastructure.
* API-based service delivery.
* Transaction-switch architectures.
* Machine-learning-assisted routing.
* Simulation-based performance evaluation.
* Digital financial services modernization.

The framework is designed for research, experimentation, benchmarking, and educational purposes.

Production deployment would require additional validation, including:

* Real-world operational telemetry.
* Security and access controls.
* Reliability and resilience testing.
* Capacity and scalability assessment.
* Model governance and monitoring.
* Institutional and regulatory compliance.
* Integration with existing banking infrastructure.

## Repository Maintenance and Release

To align the repository with the manuscript version, the following maintenance activities are required:

1. Update the README to reference `run_all.py` as the primary experimental entry point.
2. Commit the accompanying `requirements.txt` file to the repository.
3. Create a tagged GitHub release corresponding to the manuscript version.

These activities require repository write access. If write access is unavailable, the repository owner or an authorized collaborator must perform the commit and release operations.

## License

Add the applicable project license here if one has been selected.

## Citation

If you use this repository or build upon its methodology, please cite the associated research manuscript.

> Citation details should be added once the manuscript's final bibliographic information is available.

---

**Project:** Intelligent Transaction Routing in Retail Banking

**Purpose:** Research, experimentation, and educational evaluation of banking transaction-routing architectures.

