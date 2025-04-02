# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR) 

**Authors**: Mahak Agarwal, Divyam Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2504.00374)  

**Abstract**: In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation. 

---
# JudgeLRM: Large Reasoning Models as a Judge 

**Authors**: Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00050)  

**Abstract**: The rise of Large Language Models (LLMs) as evaluators offers a scalable alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for judges approaches often fall short in domains requiring complex reasoning. In this work, we investigate whether LLM judges truly benefit from enhanced reasoning capabilities. Through a detailed analysis of reasoning requirements across evaluation tasks, we reveal a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples - highlighting the limitations of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM models consistently outperform both SFT-tuned and state-of-the-art reasoning models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks requiring deep reasoning. 

---
# AI Judges in Design: Statistical Perspectives on Achieving Human Expert Equivalence With Vision-Language Models 

**Authors**: Kristen M. Edwards, Farnaz Tehranchi, Scarlett R. Miller, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00938)  

**Abstract**: The subjective evaluation of early stage engineering designs, such as conceptual sketches, traditionally relies on human experts. However, expert evaluations are time-consuming, expensive, and sometimes inconsistent. Recent advances in vision-language models (VLMs) offer the potential to automate design assessments, but it is crucial to ensure that these AI ``judges'' perform on par with human experts. However, no existing framework assesses expert equivalence. This paper introduces a rigorous statistical framework to determine whether an AI judge's ratings match those of human experts. We apply this framework in a case study evaluating four VLM-based judges on key design metrics (uniqueness, creativity, usefulness, and drawing quality). These AI judges employ various in-context learning (ICL) techniques, including uni- vs. multimodal prompts and inference-time reasoning. The same statistical framework is used to assess three trained novices for expert-equivalence. Results show that the top-performing AI judge, using text- and image-based ICL with reasoning, achieves expert-level agreement for uniqueness and drawing quality and outperforms or matches trained novices across all metrics. In 6/6 runs for both uniqueness and creativity, and 5/6 runs for both drawing quality and usefulness, its agreement with experts meets or exceeds that of the majority of trained novices. These findings suggest that reasoning-supported VLM models can achieve human-expert equivalence in design evaluation. This has implications for scaling design evaluation in education and practice, and provides a general statistical framework for validating AI judges in other domains requiring subjective content evaluation. 

---
# Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices 

**Authors**: Xiao Yan, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.00002)  

**Abstract**: Recent advancements in large language models (LLMs) have prompted interest in deploying these models on mobile devices to enable new applications without relying on cloud connectivity. However, the efficiency constraints of deploying LLMs on resource-limited devices present significant challenges. In this paper, we conduct a comprehensive measurement study to evaluate the efficiency tradeoffs between mobile-based, edge-based, and cloud-based deployments for LLM applications. We implement AutoLife-Lite, a simplified LLM-based application that analyzes smartphone sensor data to infer user location and activity contexts. Our experiments reveal that: (1) Only small-size LLMs (<4B parameters) can run successfully on powerful mobile devices, though they exhibit quality limitations compared to larger models; (2) Model compression is effective in lower the hardware requirement, but may lead to significant performance degradation; (3) The latency to run LLMs on mobile devices with meaningful output is significant (>30 seconds), while cloud services demonstrate better time efficiency (<10 seconds); (4) Edge deployments offer intermediate tradeoffs between latency and model capabilities, with different results on CPU-based and GPU-based settings. These findings provide valuable insights for system designers on the current limitations and future directions for on-device LLM applications. 

---
