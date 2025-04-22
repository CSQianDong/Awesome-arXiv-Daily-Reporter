# Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges 

**Authors**: Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15205)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to generate answers with citations from source documents containing "ground truth", thereby reducing system hallucinations. A crucial factor in RAG evaluation is "support", whether the information in the cited documents supports the answer. To this end, we conducted a large-scale comparative study of 45 participant submissions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM judge (GPT-4o) against human judges for support assessment. We considered two conditions: (1) fully manual assessments from scratch and (2) manual assessments with post-editing of LLM predictions. Our results indicate that for 56% of the manual from-scratch assessments, human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the manual with post-editing condition. Furthermore, by carefully analyzing the disagreements in an unbiased study, we found that an independent human judge correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable alternative for support assessment. To conclude, we provide a qualitative analysis of human and GPT-4o errors to help guide future iterations of support assessment. 

---
# Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators 

**Authors**: Yilun Zhou, Austin Xu, Peifeng Wang, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.15253)  

**Abstract**: Scaling test-time computation, or affording a generator large language model (LLM) extra compute during inference, typically employs the help of external non-generative evaluators (i.e., reward models). Concurrently, LLM-judges, models trained to generate evaluations and critiques (explanations) in natural language, are becoming increasingly popular in automatic evaluation. Despite judge empirical successes, their effectiveness as evaluators in test-time scaling settings is largely unknown. In this paper, we introduce the Judge Evaluation for Test-Time Scaling (JETTS) benchmark, which evaluates judge performance in three domains (math reasoning, code generation, and instruction following) under three task settings: response reranking, step-level beam search, and critique-based response refinement. We evaluate 10 different judge models (7B-70B parameters) for 8 different base generator models (6.7B-72B parameters). Our benchmark shows that while judges are competitive with outcome reward models in reranking, they are consistently worse than process reward models in beam search procedures. Furthermore, though unique to LLM-judges, their natural language critiques are currently ineffective in guiding the generator towards better responses. 

---
# Human aversion? Do AI Agents Judge Identity More Harshly Than Performance 

**Authors**: Yuanjun Feng, Vivek Chodhary, Yash Raj Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2504.13871)  

**Abstract**: This study examines the understudied role of algorithmic evaluation of human judgment in hybrid decision-making systems, a critical gap in management research. While extant literature focuses on human reluctance to follow algorithmic advice, we reverse the perspective by investigating how AI agents based on large language models (LLMs) assess and integrate human input. Our work addresses a pressing managerial constraint: firms barred from deploying LLMs directly due to privacy concerns can still leverage them as mediating tools (for instance, anonymized outputs or decision pipelines) to guide high-stakes choices like pricing or discounts without exposing proprietary data. Through a controlled prediction task, we analyze how an LLM-based AI agent weights human versus algorithmic predictions. We find that the AI system systematically discounts human advice, penalizing human errors more severely than algorithmic errors--a bias exacerbated when the agent's identity (human vs AI) is disclosed and the human is positioned second. These results reveal a disconnect between AI-generated trust metrics and the actual influence of human judgment, challenging assumptions about equitable human-AI collaboration. Our findings offer three key contributions. First, we identify a reverse algorithm aversion phenomenon, where AI agents undervalue human input despite comparable error rates. Second, we demonstrate how disclosure and positional bias interact to amplify this effect, with implications for system design. Third, we provide a framework for indirect LLM deployment that balances predictive power with data privacy. For practitioners, this research emphasize the need to audit AI weighting mechanisms, calibrate trust dynamics, and strategically design decision sequences in human-AI systems. 

---
