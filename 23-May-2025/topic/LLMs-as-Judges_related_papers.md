# SLMEval: Entropy-Based Calibration for Human-Aligned Evaluation of Large Language Models 

**Authors**: Roland Daynauth, Christopher Clarke, Krisztian Flautner, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2505.16003)  

**Abstract**: The LLM-as-a-Judge paradigm offers a scalable, reference-free approach for evaluating language models. Although several calibration techniques have been proposed to better align these evaluators with human judgment, prior studies focus primarily on narrow, well-structured benchmarks. As a result, it remains unclear whether such calibrations generalize to real-world, open-ended tasks.
In this work, we show that SOTA calibrated evaluators often fail in these settings, exhibiting weak or even negative correlation with human judgments. To address this, we propose SLMEval, a novel and efficient calibration method based on entropy maximization over a small amount of human preference data. By estimating a latent distribution over model quality and reweighting evaluator scores accordingly, SLMEval achieves strong correlation with human evaluations across two real-world production use cases and the public benchmark. For example, on one such task, SLMEval achieves a Spearman correlation of 0.57 with human judgments, while G-Eval yields a negative correlation. In addition, SLMEval reduces evaluation costs by 5-30x compared to GPT-4-based calibrated evaluators such as G-eval. 

---
# Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation 

**Authors**: Jiwon Moon, Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.16222)  

**Abstract**: With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to both positive and negative biases, resulting in inflated or unfairly low scores. Moreover, we observe that LLM judges remain vulnerable to these biases even when prompted to generate test cases before scoring, highlighting the need for more robust code evaluation methods. 

---
# OpenEthics: A Comprehensive Ethical Evaluation of Open-Source Generative Large Language Models 

**Authors**: Burak Erinç Çetin, Yıldırım Özen, Elif Naz Demiryılmaz, Kaan Engür, Cagri Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2505.16036)  

**Abstract**: Generative large language models present significant potential but also raise critical ethical concerns. Most studies focus on narrow ethical dimensions, and also limited diversity of languages and models. To address these gaps, we conduct a broad ethical evaluation of 29 recent open-source large language models using a novel data collection including four ethical aspects: Robustness, reliability, safety, and fairness. We analyze model behavior in both a commonly used language, English, and a low-resource language, Turkish. Our aim is to provide a comprehensive ethical assessment and guide safer model development by filling existing gaps in evaluation breadth, language coverage, and model diversity. Our experimental results, based on LLM-as-a-Judge, reveal that optimization efforts for many open-source models appear to have prioritized safety and fairness, and demonstrated good robustness while reliability remains a concern. We demonstrate that ethical evaluation can be effectively conducted independently of the language used. In addition, models with larger parameter counts tend to exhibit better ethical performance, with Gemma and Qwen models demonstrating the most ethical behavior among those evaluated. 

---
