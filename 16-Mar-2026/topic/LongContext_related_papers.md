# Developing and evaluating a chatbot to support maternal health care 

**Authors**: Smriti Jha, Vidhi Jain, Jianyu Xu, Grace Liu, Sowmya Ramesh, Jitender Nagpal, Gretchen Chapman, Benjamin Bellows, Siddhartha Goyal, Aarti Singh, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2603.13168)  

**Abstract**: The ability to provide trustworthy maternal health information using phone-based chatbots can have a significant impact, particularly in low-resource settings where users have low health literacy and limited access to care. However, deploying such systems is technically challenging: user queries are short, underspecified, and code-mixed across languages, answers require regional context-specific grounding, and partial or missing symptom context makes safe routing decisions difficult.
We present a chatbot for maternal health in India developed through a partnership between academic researchers, a health tech company, a public health nonprofit, and a hospital. The system combines (1) stage-aware triage, routing high-risk queries to expert templates, (2) hybrid retrieval over curated maternal/newborn guidelines, and (3) evidence-conditioned generation from an LLM. Our core contribution is an evaluation workflow for high-stakes deployment under limited expert supervision. Targeting both component-level and end-to-end testing, we introduce: (i) a labeled triage benchmark (N=150) achieving 86.7% emergency recall, explicitly reporting the missed-emergency vs. over-escalation trade-off; (ii) a synthetic multi-evidence retrieval benchmark (N=100) with chunk-level evidence labels; (iii) LLM-as-judge comparison on real queries (N=781) using clinician-codesigned criteria; and (iv) expert validation. Our findings show that trustworthy medical assistants in multilingual, noisy settings require defense-in-depth design paired with multi-method evaluation, rather than any single model and evaluation method choice. 

---
# ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation 

**Authors**: Siqi Sun, Ben Peng Wu, Mali Jin, Peizhen Bai, Hanpei Zhang, Xingyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.13154)  

**Abstract**: As corporate responsibility increasingly incorporates environmental, social, and governance (ESG) criteria, ESG reporting is becoming a legal requirement in many regions and a key channel for documenting sustainability practices and assessing firms' long-term and ethical performance. However, the length and complexity of ESG disclosures make them difficult to interpret and automate the analysis reliably. To support scalable and trustworthy analysis, this paper introduces ESG-Bench, a benchmark dataset for ESG report understanding and hallucination mitigation in large language models (LLMs). ESG-Bench contains human-annotated question-answer (QA) pairs grounded in real-world ESG report contexts, with fine-grained labels indicating whether model outputs are factually supported or hallucinated. Framing ESG report analysis as a QA task with verifiability constraints enables systematic evaluation of LLMs' ability to extract and reason over ESG content and provides a new use case: mitigating hallucinations in socially sensitive, compliance-critical settings. We design task-specific Chain-of-Thought (CoT) prompting strategies and fine-tune multiple state-of-the-art LLMs on ESG-Bench using CoT-annotated rationales. Our experiments show that these CoT-based methods substantially outperform standard prompting and direct fine-tuning in reducing hallucinations, and that the gains transfer to existing QA benchmarks beyond the ESG domain. 

---
# Long-form RewardBench: Evaluating Reward Models for Long-form Generation 

**Authors**: Hui Huang, Yancheng He, Wei Liu, Muyun Yang, Jiaheng Liu, Kehai Chen, Bing Xu, Conghui Zhu, Hailong Cao, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.12963)  

**Abstract**: The widespread adoption of reinforcement learning-based alignment highlights the growing importance of reward models. Various benchmarks have been built to evaluate reward models in various domains and scenarios. However, a significant gap remains in assessing reward models for long-form generation, despite its critical role in real-world applications. To bridge this, we introduce Long-form RewardBench, the first reward modeling testbed specifically designed for long-form generation. Our benchmark encompasses five key subtasks: QA, RAG, Chat, Writing, and Reasoning. We collected instruction and preference data through a meticulously designed multi-stage data collection process, and conducted extensive experiments on 20+ mainstream reward models, including both classifiers and generative models. Our findings reveal that current models still lack long-form reward modeling capabilities. Furthermore, we designed a novel Long-form Needle-in-a-Haystack Test, which revealed a correlation between reward modeling performance and the error's position within a response, as well as the overall response length, with distinct characteristics observed between classification and generative models. Finally, we demonstrate that classifiers exhibit better generalizability compared to generative models trained on the same data. As the first benchmark for long-form reward modeling, this work aims to offer a robust platform for visualizing progress in this crucial area. 

---
# Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates in Large Language Models 

**Authors**: Boyu Qiao, Sean Guo, Xian Yang, Kun Li, Wei Zhou, Songlin Hu, Yunya Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.12271)  

**Abstract**: LLMs are widely used in knowledge-intensive tasks where the same fact may be revised multiple times within context. Unlike prior work focusing on one-shot updates or single conflicts, multi-update scenarios contain multiple historically valid versions that compete at retrieval, yet remain underexplored. This challenge resembles the AB-AC interference paradigm in cognitive psychology: when the same cue A is successively associated with B and C, the old and new associations compete during retrieval, leading to bias. Inspired by this, we introduce a Dynamic Knowledge Instance (DKI) evaluation framework, modeling multi-updates of the same fact as a cue paired with a sequence of updated values, and assess models via endpoint probing of the earliest (initial) and latest (current) states. Across diverse LLMs, we observe that retrieval bias intensifies as updates increase, earliest-state accuracy stays high while latest-state accuracy drops substantially. Diagnostic analyses of attention, hidden-state similarity, and output logits further reveal that these signals become flatter and weakly discriminative on errors, providing little stable basis for identifying the latest update. Finally, cognitively inspired heuristic intervention strategies yield only modest gains and do not eliminate the bias. Our results reveal a persistent challenge in tracking and following knowledge updates in long contexts. 

---
