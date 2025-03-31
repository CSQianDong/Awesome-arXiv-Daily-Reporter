# ELM: Ensemble of Language Models for Predicting Tumor Group from Pathology Reports 

**Authors**: Lovedeep Gondara, Jonathan Simkin, Shebnum Devji, Gregory Arbour, Raymond Ng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21800)  

**Abstract**: Population-based cancer registries (PBCRs) face a significant bottleneck in manually extracting data from unstructured pathology reports, a process crucial for tasks like tumor group assignment, which can consume 900 person-hours for approximately 100,000 reports. To address this, we introduce ELM (Ensemble of Language Models), a novel ensemble-based approach leveraging both small language models (SLMs) and large language models (LLMs). ELM utilizes six fine-tuned SLMs, where three SLMs use the top part of the pathology report and three SLMs use the bottom part. This is done to maximize report coverage. ELM requires five-out-of-six agreement for a tumor group classification. Disagreements are arbitrated by an LLM with a carefully curated prompt. Our evaluation across nineteen tumor groups demonstrates ELM achieves an average precision and recall of 0.94, outperforming single-model and ensemble-without-LLM approaches. Deployed at the British Columbia Cancer Registry, ELM demonstrates how LLMs can be successfully applied in a PBCR setting to achieve state-of-the-art results and significantly enhance operational efficiencies, saving hundreds of person-hours annually. 

---
# Debate-Driven Multi-Agent LLMs for Phishing Email Detection 

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22038)  

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies. 

---
