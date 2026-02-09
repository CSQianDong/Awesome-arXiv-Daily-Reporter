# MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs 

**Authors**: Junhyeok Lee, Han Jang, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06268)  

**Abstract**: Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly integrated into clinical workflows; however, prompt injection attacks can steer these systems toward clinically unsafe or misleading outputs. We introduce the Medical Prompt Injection Benchmark (MPIB), a dataset-and-benchmark suite for evaluating clinical safety under both direct prompt injection and indirect, RAG-mediated injection across clinically grounded tasks. MPIB emphasizes outcome-level risk via the Clinical Harm Event Rate (CHER), which measures high-severity clinical harm events under a clinically grounded taxonomy, and reports CHER alongside Attack Success Rate (ASR) to disentangle instruction compliance from downstream patient risk. The benchmark comprises 9,697 curated instances constructed through multi-stage quality gates and clinical safety linting. Evaluating MPIB across a diverse set of baseline LLMs and defense configurations, we find that ASR and CHER can diverge substantially, and that robustness depends critically on whether adversarial instructions appear in the user query or in retrieved context. We release MPIB with evaluation code, adversarial baselines, and comprehensive documentation to support reproducible and systematic research on clinical prompt injection. Code and data are available at GitHub (code) and Hugging Face (data). 

---
