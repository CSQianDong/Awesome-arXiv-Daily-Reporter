# RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition 

**Authors**: Kun Ran, Marwah Alaofi, Danula Hettiachchi, Chenglong Ma, Khoi Nguyen Dinh Anh, Khoi Vo Nguyen, Sachin Pathiyan Cherumanal, Lida Rashidi, Falk Scholer, Damiano Spina, Shuoqi Sun, Oleg Zendel  

**Link**: [PDF](https://arxiv.org/pdf/2602.20735)  

**Abstract**: This paper presents the award-winning RMIT-ADM+S system for the Text-to-Text
track of the NeurIPS~2025 MMU-RAG Competition. We introduce Routing-to-RAG
(R2RAG), a research-focused retrieval-augmented generation (RAG)
architecture composed of lightweight components that dynamically adapt the
retrieval strategy based on inferred query complexity and evidence
sufficiency. The system uses smaller LLMs, enabling operation on a single
consumer-grade GPU while supporting complex research tasks. It builds on the
G-RAG system, winner of the ACM~SIGIR~2025 LiveRAG Challenge, and extends it
with modules informed by qualitative review of outputs. R2RAG won the Best
Dynamic Evaluation award in the Open Source category, demonstrating high
effectiveness with careful design and efficient use of resources. 

---
# Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems 

**Authors**: Mukul Chhabra, Luigi Medrano, Arush Verma  

**Link**: [PDF](https://arxiv.org/pdf/2602.20379)  

**Abstract**: Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows such as technical support and IT operations, where evaluation must reflect operational constraints, structured identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific failure modes such as case misidentification, workflow misalignment, and partial resolution across turns.
We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems. The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality, grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation, regression testing, and production monitoring.
Through a comparative study of two instruction-tuned models across short and long workflows, we show that generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical tradeoffs that are actionable for system improvement. 

---
