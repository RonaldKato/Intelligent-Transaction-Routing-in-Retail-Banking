Intelligent Transaction Routing in Retail Banking

<img width="100%" alt="pipeline" src="https://github.com/user-attachments/assets/3d5c1ca7-f489-43a9-b2b7-124eab72d252" />

A reproducible, simulation-based framework for evaluating transaction-routing architectures in retail banking. The project compares traditional Transaction Switch, API Gateway, and Machine-Learning Hybrid (ML-Hybrid) routing using synthetic, industry-parameterized banking transactions.

**Overview**
The framework investigates whether API-based routing provides measurable performance advantages over traditional switch-based architectures and whether machine-learning-based routing can provide additional optimization when both routing options are available.
The simulation uses 60,000 synthetic transactions across five banking channels:
POS, ATM, E-commerce, Mobile Application, USSD
It models six transaction types: PURCHASE, WITHDRAWAL, BALANCE_INQUIRY, FUND_TRANSFER, BILL_PAYMENT, and REVERSAL.

**Routing Strategies**
The framework evaluates three routing approaches:
  Switch-only — routes all transactions through the traditional Transaction Switch.
  API-only — routes all transactions through the API Gateway.
  ML-Hybrid — uses a machine-learning router to select between the Transaction Switch and API Gateway.
  Evaluation Metrics

**The framework evaluates:**
  Transaction latency, Throughput capacity, Transaction success rate, Routing decisions, Workload sensitivity, Statistical significance, Effect size, Bootstrap confidence intervals, Machine-learning classification performance, Feature importance, Methodology
  A coordinated Python pipeline generates and validates the synthetic workload, engineers routing features, executes the architectural simulations under identical conditions, trains and tunes the ML router, and performs statistical analysis.
  Paired statistical procedures are used to compare architectural performance while accounting for the paired transaction structure. The analysis includes paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d$, and bootstrap confidence intervals.

**Key Findings**
Under the modeled conditions:
  API Gateway latency: 199.60 ms
  Transaction Switch latency: 255.35 ms
  Latency reduction: 21.83%
  API throughput proxy: 226.5 transactions/s
  Switch throughput proxy: 177.0 transactions/s
  Throughput improvement: approximately 28.0%
  Transaction success rates: statistically comparable
  ML-Hybrid improvement over API-only: approximately 0.15%
  Transactions routed to API by ML-Hybrid: approximately 99.5%

The results indicate that architectural modernization provides the dominant performance improvement, while machine-learning routing offers limited additional benefit when the API architecture already performs substantially better.

**Clone the repository:**
git clone https://github.com/RonaldKato/Intelligent-Transaction-Routing-in-Retail-Banking.git
cd Intelligent-Transaction-Routing-in-Retail-Banking

**Create a virtual environment:**
python -m venv venv

**Activate the environment:**
Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate

**Install dependencies:**
pip install -r requirements.txt

**Run the experimental pipeline:**
python pipeline.py
Reproducibility

The experiments use parameterized synthetic data and controlled simulation conditions to support reproducibility. The dataset does not represent production banking traffic and should not be interpreted as a direct measurement of live infrastructure performance.

**Ethical Considerations**
The study uses fully synthetic transaction data. No customer information, personally identifiable information, production banking records, or confidential institutional data are included.

**Research Context**
This repository supports research on modern banking transaction infrastructure, API-based service delivery, transaction-switch architectures, and machine-learning-assisted routing.

The framework is intended for research, experimentation, benchmarking, and educational purposes. Production deployment would require validation using appropriate operational telemetry, security controls, reliability testing, governance processes, and institutional requirements.
