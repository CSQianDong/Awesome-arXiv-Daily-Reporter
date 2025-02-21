# Investigating Non-Transitivity in LLM-as-a-Judge 

**Title (ZH)**: 探究LLM作为法官时的非传递性现象 

**Authors**: Yi Xu, Laura Ruis, Tim Rocktäschel, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2502.14074)  

**Abstract**: Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的自动评估方法正在成为评估LLM驱动代理的指令遵循能力的标准工具。这一范式中最常见的方法是使用基线模型进行成对比较，这种方法严重依赖于传递偏好假设。然而，这一假设的有效性尚未得到充分探索。本研究旨在调查AlpacaEval框架内的非传递性现象及其对模型排名的影响。我们发现，LLM评判者表现出非传递性偏好，导致排名对基线模型的选择非常敏感。为缓解这一问题，我们证明了循环赛结合布雷德利-泰利（Bradley-Terry）偏好模型可以产生更可靠的排名。值得注意的是，我们的方法提高了与Chatbot Arena的斯皮尔曼等级相关性和肯德尔等级相关性（分别为95.0% -> 96.4% 和82.1% -> 86.3%）。为应对循环赛的计算成本，我们提出了“智胜”循环赛（Swim）方法，通过动态匹配策略同时捕捉循环赛的优势和保持计算效率。 

---
