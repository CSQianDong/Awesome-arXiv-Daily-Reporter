# HyperCLOVA X THINK Technical Report 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2506.22403)  

**Abstract**: We introduce HyperCLOVA X THINK, the first reasoning-focused large language model in the HyperCLOVA X family, pre-trained on roughly $6$ trillion high-quality Korean, and English tokens, augmented with targeted synthetic Korean data. It was implemented as a compute-memory-balanced Peri-LN Transformer scaled with $\mu$P, pre-trained through a three-stage curriculum that expands the context window to $128$K tokens, and post-trained via supervised fine-tuning with Reinforcement Learning from Verifiable Rewards supports both detailed rationale and concise-answer modes. It delivers competitive performance against similarly sized models on Korea-focused benchmarks such as KMMLU, CSAT, KoBALT-700, HAERAE-1.0, and KoBigBench, while preserving robust bilingual consistency and translation quality. In addition, a vision-augmented variant matches or exceeds GPT-4.1 on the KCSAT STEM benchmark, all of which are achieved with substantially lower training compute than existing models of similar sizes. We also present a pruning and distillation technique that will soon be applied to HyperCLOVA X THINK for an open-source and business-friendly foundation model. Altogether, these capabilities position HyperCLOVA X THINK as a robust foundation for Korean AI innovation and a valuable resource for the global research community. 

---
# ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation 

**Authors**: Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21931)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown promise in enhancing recommendation systems by incorporating external context into large language model prompts. However, existing RAG-based approaches often rely on static retrieval heuristics and fail to capture nuanced user preferences in dynamic recommendation scenarios. In this work, we introduce ARAG, an Agentic Retrieval-Augmented Generation framework for Personalized Recommendation, which integrates a multi-agent collaboration mechanism into the RAG pipeline. To better understand the long-term and session behavior of the user, ARAG leverages four specialized LLM-based agents: a User Understanding Agent that summarizes user preferences from long-term and session contexts, a Natural Language Inference (NLI) Agent that evaluates semantic alignment between candidate items retrieved by RAG and inferred intent, a context summary agent that summarizes the findings of NLI agent, and an Item Ranker Agent that generates a ranked list of recommendations based on contextual fit. We evaluate ARAG accross three datasets. Experimental results demonstrate that ARAG significantly outperforms standard RAG and recency-based baselines, achieving up to 42.1% improvement in NDCG@5 and 35.5% in Hit@5. We also, conduct an ablation study to analyse the effect by different components of ARAG. Our findings highlight the effectiveness of integrating agentic reasoning into retrieval-augmented recommendation and provide new directions for LLM-based personalization. 

---
# Structured Attention Matters to Multimodal LLMs in Document Understanding 

**Authors**: Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21600)  

**Abstract**: Document understanding remains a significant challenge for multimodal large language models (MLLMs). While previous research has primarily focused on locating evidence pages through precise multimodal queries, our work investigates a fundamental yet overlooked aspect: how input format influences document comprehension performance. Through systematic analysis, we discover that raw OCR text often impairs rather than improves MLLMs' performance, which is a counterintuitive finding we attribute to attention dispersion and structure loss. To further substantiate our hypothesis, we propose a novel structure-preserving approach that encodes document elements using the LaTex paradigm, maintaining the hierarchical organization and spatial relationships critical for comprehension. Our attention analysis reveals that structured text induces structured attention patterns on both textual and visual content, directing models to focus on semantically meaningful regions while reducing attention waste. This approach significantly enhances MLLMs' document question answering performance across diverse document types without requiring architectural modifications or additional training. 

---
# PEACE: Empowering Geologic Map Holistic Understanding with MLLMs 

**Authors**: Yangyu Huang, Tianyi Gao, Haoran Xu, Qihao Zhao, Yang Song, Zhipeng Gui, Tengchao Lv, Hao Chen, Lei Cui, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.06184)  

**Abstract**: Geologic map, as a fundamental diagram in geology science, provides critical insights into the structure and composition of Earth's subsurface and surface. These maps are indispensable in various fields, including disaster detection, resource exploration, and civil engineering. Despite their significance, current Multimodal Large Language Models (MLLMs) often fall short in geologic map understanding. This gap is primarily due to the challenging nature of cartographic generalization, which involves handling high-resolution map, managing multiple associated components, and requiring domain-specific knowledge. To quantify this gap, we construct GeoMap-Bench, the first-ever benchmark for evaluating MLLMs in geologic map understanding, which assesses the full-scale abilities in extracting, referring, grounding, reasoning, and analyzing. To bridge this gap, we introduce GeoMap-Agent, the inaugural agent designed for geologic map understanding, which features three modules: Hierarchical Information Extraction (HIE), Domain Knowledge Injection (DKI), and Prompt-enhanced Question Answering (PEQA). Inspired by the interdisciplinary collaboration among human scientists, an AI expert group acts as consultants, utilizing a diverse tool pool to comprehensively analyze questions. Through comprehensive experiments, GeoMap-Agent achieves an overall score of 0.811 on GeoMap-Bench, significantly outperforming 0.369 of GPT-4o. Our work, emPowering gEologic mAp holistiC undErstanding (PEACE) with MLLMs, paves the way for advanced AI applications in geology, enhancing the efficiency and accuracy of geological investigations. 

---
# Refine Medical Diagnosis Using Generation Augmented Retrieval and Clinical Practice Guidelines 

**Authors**: Wenhao Li, Hongkuan Zhang, Hongwei Zhang, Zhengxu Li, Zengjie Dong, Yafan Chen, Niranjan Bidargaddi, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21615)  

**Abstract**: Current medical language models, adapted from large language models (LLMs), typically predict ICD code-based diagnosis from electronic health records (EHRs) because these labels are readily available. However, ICD codes do not capture the nuanced, context-rich reasoning clinicians use for diagnosis. Clinicians synthesize diverse patient data and reference clinical practice guidelines (CPGs) to make evidence-based decisions. This misalignment limits the clinical utility of existing models. We introduce GARMLE-G, a Generation-Augmented Retrieval framework that grounds medical language model outputs in authoritative CPGs. Unlike conventional Retrieval-Augmented Generation based approaches, GARMLE-G enables hallucination-free outputs by directly retrieving authoritative guideline content without relying on model-generated text. It (1) integrates LLM predictions with EHR data to create semantically rich queries, (2) retrieves relevant CPG knowledge snippets via embedding similarity, and (3) fuses guideline content with model output to generate clinically aligned recommendations. A prototype system for hypertension diagnosis was developed and evaluated on multiple metrics, demonstrating superior retrieval precision, semantic relevance, and clinical guideline adherence compared to RAG-based baselines, while maintaining a lightweight architecture suitable for localized healthcare deployment. This work provides a scalable, low-cost, and hallucination-free method for grounding medical language models in evidence-based clinical practice, with strong potential for broader clinical deployment. 

---
