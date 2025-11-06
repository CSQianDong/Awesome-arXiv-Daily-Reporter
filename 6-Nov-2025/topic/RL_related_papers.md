# Knowledge-Augmented Question Error Correction for Chinese Question Answer System with QuestionRAG 

**Authors**: Longpeng Qiu, Ting Li, Shuai Mao, Nan Yang, Xiaohui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03410)  

**Abstract**: Input errors in question-answering (QA) systems often lead to incorrect responses. Large language models (LLMs) struggle with this task, frequently failing to interpret user intent (misinterpretation) or unnecessarily altering the original question's structure (over-correction). We propose QuestionRAG, a framework that tackles these problems. To address misinterpretation, it enriches the input with external knowledge (e.g., search results, related entities). To prevent over-correction, it uses reinforcement learning (RL) to align the model's objective with precise correction, not just paraphrasing. Our results demonstrate that knowledge augmentation is critical for understanding faulty questions. Furthermore, RL-based alignment proves significantly more effective than traditional supervised fine-tuning (SFT), boosting the model's ability to follow instructions and generalize. By integrating these two strategies, QuestionRAG unlocks the full potential of LLMs for the question correction task. 

---
