# Long-context Reference-based MT Quality Estimation 

**Authors**: Sami Ul Haq, Chinonso Cynthia Osuji, Sheila Castilho, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2509.13980)  

**Abstract**: In this paper, we present our submission to the Tenth Conference on Machine Translation (WMT25) Shared Task on Automated Translation Quality Evaluation.
Our systems are built upon the COMET framework and trained to predict segment-level Error Span Annotation (ESA) scores using augmented long-context data.
To construct long-context training data, we concatenate in-domain, human-annotated sentences and compute a weighted average of their scores.
We integrate multiple human judgment datasets (MQM, SQM, and DA) by normalising their scales and train multilingual regression models to predict quality scores from the source, hypothesis, and reference translations.
Experimental results show that incorporating long-context information improves correlations with human judgments compared to models trained only on short segments. 

---
# Improving Context Fidelity via Native Retrieval-Augmented Reasoning 

**Authors**: Suyuchen Wang, Jinlin Wang, Xinyu Wang, Shiqi Li, Xiangru Tang, Sirui Hong, Xiao-Wen Chang, Chenglin Wu, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13683)  

**Abstract**: Large language models (LLMs) often struggle with context fidelity, producing inconsistent answers when responding to questions based on provided information. Existing approaches either rely on expensive supervised fine-tuning to generate evidence post-answer or train models to perform web searches without necessarily improving utilization of the given context. We propose CARE, a novel native retrieval-augmented reasoning framework that teaches LLMs to explicitly integrate in-context evidence within their reasoning process with the model's own retrieval capabilities. Our method requires limited labeled evidence data while significantly enhancing both retrieval accuracy and answer generation performance through strategically retrieved in-context tokens in the reasoning chain. Extensive experiments on multiple real-world and counterfactual QA benchmarks demonstrate that our approach substantially outperforms supervised fine-tuning, traditional retrieval-augmented generation methods, and external retrieval solutions. This work represents a fundamental advancement in making LLMs more accurate, reliable, and efficient for knowledge-intensive tasks. 

---
