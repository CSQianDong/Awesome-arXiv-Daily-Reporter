# Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering 

**Authors**: Xueren Ge, Sahil Murtaza, Anthony Cortez, Homa Alemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10900)  

**Abstract**: Large language models (LLMs) have shown promise in medical question answering, yet they often overlook the domain-specific expertise that professionals depend on, such as the clinical subject areas (e.g., trauma, airway) and the certification level (e.g., EMT, Paramedic). Existing approaches typically apply general-purpose prompting or retrieval strategies without leveraging this structured context, limiting performance in high-stakes settings. We address this gap with EMSQA, an 24.3K-question multiple-choice dataset spanning 10 clinical subject areas and 4 certification levels, accompanied by curated, subject area-aligned knowledge bases (40K documents and 2M tokens). Building on EMSQA, we introduce (i) Expert-CoT, a prompting strategy that conditions chain-of-thought (CoT) reasoning on specific clinical subject area and certification level, and (ii) ExpertRAG, a retrieval-augmented generation pipeline that grounds responses in subject area-aligned documents and real-world patient data. Experiments on 4 LLMs show that Expert-CoT improves up to 2.05% over vanilla CoT prompting. Additionally, combining Expert-CoT with ExpertRAG yields up to a 4.59% accuracy gain over standard RAG baselines. Notably, the 32B expertise-augmented LLMs pass all the computer-adaptive EMS certification simulation exams. 

---
# Multimodal Peer Review Simulation with Actionable To-Do Recommendations for Community-Aware Manuscript Revisions 

**Authors**: Mengze Hong, Di Jiang, Weiwei Zhao, Yawen Li, Yihang Wang, Xinyuan Luo, Yanjie Sun, Chen Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10902)  

**Abstract**: While large language models (LLMs) offer promising capabilities for automating academic workflows, existing systems for academic peer review remain constrained by text-only inputs, limited contextual grounding, and a lack of actionable feedback. In this work, we present an interactive web-based system for multimodal, community-aware peer review simulation to enable effective manuscript revisions before paper submission. Our framework integrates textual and visual information through multimodal LLMs, enhances review quality via retrieval-augmented generation (RAG) grounded in web-scale OpenReview data, and converts generated reviews into actionable to-do lists using the proposed Action:Objective[\#] format, providing structured and traceable guidance. The system integrates seamlessly into existing academic writing platforms, providing interactive interfaces for real-time feedback and revision tracking. Experimental results highlight the effectiveness of the proposed system in generating more comprehensive and useful reviews aligned with expert standards, surpassing ablated baselines and advancing transparent, human-centered scholarly assistance. 

---
# ICX360: In-Context eXplainability 360 Toolkit 

**Authors**: Dennis Wei, Ronny Luss, Xiaomeng Hu, Lucas Monteiro Paes, Pin-Yu Chen, Karthikeyan Natesan Ramamurthy, Erik Miehling, Inge Vejsbjerg, Hendrik Strobelt  

**Link**: [PDF](https://arxiv.org/pdf/2511.10879)  

**Abstract**: Large Language Models (LLMs) have become ubiquitous in everyday life and are entering higher-stakes applications ranging from summarizing meeting transcripts to answering doctors' questions. As was the case with earlier predictive models, it is crucial that we develop tools for explaining the output of LLMs, be it a summary, list, response to a question, etc. With these needs in mind, we introduce In-Context Explainability 360 (ICX360), an open-source Python toolkit for explaining LLMs with a focus on the user-provided context (or prompts in general) that are fed to the LLMs. ICX360 contains implementations for three recent tools that explain LLMs using both black-box and white-box methods (via perturbations and gradients respectively). The toolkit, available at this https URL, contains quick-start guidance materials as well as detailed tutorials covering use cases such as retrieval augmented generation, natural language generation, and jailbreaking. 

---
