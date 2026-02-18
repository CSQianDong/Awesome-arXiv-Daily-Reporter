# Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory 

**Authors**: Zihao Tang, Xin Yu, Ziyu Xiao, Zengxuan Wen, Zelin Li, Jiaxi Zhou, Hualei Wang, Haohua Wang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15313)  

**Abstract**: AI Memory, specifically how models organizes and retrieves historical messages, becomes increasingly valuable to Large Language Models (LLMs), yet existing methods (RAG and Graph-RAG) primarily retrieve memory through similarity-based mechanisms. While efficient, such System-1-style retrieval struggles with scenarios that require global reasoning or comprehensive coverage of all relevant information. In this work, We propose Mnemis, a novel memory framework that integrates System-1 similarity search with a complementary System-2 mechanism, termed Global Selection. Mnemis organizes memory into a base graph for similarity retrieval and a hierarchical graph that enables top-down, deliberate traversal over semantic hierarchies. By combining the complementary strength from both retrieval routes, Mnemis retrieves memory items that are both semantically and structurally relevant. Mnemis achieves state-of-the-art performance across all compared methods on long-term memory benchmarks, scoring 93.9 on LoCoMo and 91.6 on LongMemEval-S using GPT-4.1-mini. 

---
# GLM-5: from Vibe Coding to Agentic Engineering 

**Authors**: GLM-5 Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chen Li, Chenghua Huang, Chengwei Hu, Chenhui Zhang, Chenzheng Zhu, Congfeng Yin, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huan Liu, Huanpeng Chu, Jia'ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.15763)  

**Abstract**: We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon the agentic, reasoning, and coding (ARC) capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity. To advance model alignment and autonomy, we implement a new asynchronous reinforcement learning infrastructure that drastically improves post-training efficiency by decoupling generation from training. Furthermore, we propose novel asynchronous agent RL algorithms that further improve RL quality, enabling the model to learn from complex, long-horizon interactions more effectively. Through these innovations, GLM-5 achieves state-of-the-art performance on major open benchmarks. Most critically, GLM-5 demonstrates unprecedented capability in real-world coding tasks, surpassing previous baselines in handling end-to-end software engineering challenges. Code, models, and more information are available at this https URL. 

---
# CGRA-DeBERTa Concept Guided Residual Augmentation Transformer for Theologically Islamic Understanding 

**Authors**: Tahir Hussain, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2602.15139)  

**Abstract**: Accurate QA over classical Islamic texts remains challenging due to domain specific semantics, long context dependencies, and concept sensitive reasoning. Therefore, a new CGRA DeBERTa, a concept guided residual domain augmentation transformer framework, is proposed that enhances theological QA over Hadith corpora. The CGRA DeBERTa builds on a customized DeBERTa transformer backbone with lightweight LoRA based adaptations and a residual concept aware gating mechanism. The customized DeBERTa embedding block learns global and positional context, while Concept Guided Residual Blocks incorporate theological priors from a curated Islamic Concept Dictionary of 12 core terms. Moreover, the Concept Gating Mechanism selectively amplifies semantically critical tokens via importance weighted attention, applying differential scaling from 1.04 to 3.00. This design preserves contextual integrity, strengthens domain-specific semantic representations, and enables accurate, efficient span extraction while maintaining computational efficiency. This paper reports the results of training CGRA using a specially constructed dataset of 42591 QA pairs from the text of Sahih alBukhari and Sahih Muslim. While BERT achieved an EM score of 75.87 and DeBERTa one of 89.77, our model scored 97.85 and thus surpassed them by 8.08 on an absolute scale, all while adding approximately 8 inference overhead due to parameter efficient gating. The qualitative evaluation noted better extraction and discrimination and theological precision. This study presents Hadith QA systems that are efficient, interpretable, and accurate and that scale provide educational materials with necessary theological nuance. 

---
# How to Train Your Long-Context Visual Document Model 

**Authors**: Austin Veselka  

**Link**: [PDF](https://arxiv.org/pdf/2602.15257)  

**Abstract**: We present the first comprehensive, large-scale study of training long-context vision language models up to 344K context, targeting long-document visual question answering with measured transfer to long-context text. While several such strong are open-weight, namely Qwen3 VL and GLM 4.5/6V, their training recipes and data pipelines are not reproducible. We systematically study continued pretraining, supervised finetuning, and preference optimization for 24B and 32B parameter models, backed by extensive LC evaluations and ablations to bridge this gap, and achieve state-of-the-art performance on MMLongBenchDoc for both parameter scales. In addition to this, our key findings include: (i) training on context lengths that match evaluation context lengths outperforms training on longer contexts, (ii) training and evaluating with page indices provides a simple, high-impact boost to long-document performance, (iii) our synthetic data pipelines enable self-improvement via continued pretraining and supervised finetuning, and (iv) we extend the known text-to-visual long context transfer to the reverse, showing that visual long context training transfers to long-context text performance. We also release MMLBD-C, a manually corrected version of MMLongBenchDoc to reduce erroneous and low quality examples in the benchmark. 

---
# Seeing to Generalize: How Visual Data Corrects Binding Shortcuts 

**Authors**: Nicolas Buzeta, Felipe del Rio, Cristian Hinostroza, Denis Parra, Hans Lobel, Rodrigo Toro Icarte  

**Link**: [PDF](https://arxiv.org/pdf/2602.15183)  

**Abstract**: Vision Language Models (VLMs) are designed to extend Large Language Models (LLMs) with visual capabilities, yet in this work we observe a surprising phenomenon: VLMs can outperform their underlying LLMs on purely text-only tasks, particularly in long-context information retrieval. To investigate this effect, we build a controlled synthetic retrieval task and find that a transformer trained only on text achieves perfect in-distribution accuracy but fails to generalize out of distribution, while subsequent training on an image-tokenized version of the same task nearly doubles text-only OOD performance. Mechanistic interpretability reveals that visual training changes the model's internal binding strategy: text-only training encourages positional shortcuts, whereas image-based training disrupts them through spatial translation invariance, forcing the model to adopt a more robust symbolic binding mechanism that persists even after text-only examples are reintroduced. We further characterize how binding strategies vary across training regimes, visual encoders, and initializations, and show that analogous shifts occur during pretrained LLM-to-VLM transitions. Our findings suggest that cross-modal training can enhance reasoning and generalization even for tasks grounded in a single modality. 

---
# Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs 

**Authors**: Libo Zhang, Zhaoning Zhang, Wangyang Hong, Peng Qiao, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.15318)  

**Abstract**: Although speculative decoding is widely used to accelerate Vision-Language Models (VLMs) inference, it faces severe performance collapse when applied to Video Large Language Models (Vid-LLMs). The draft model typically falls into the trap of attention dilution and negative visual gain due to key-value cache explosion and context window mismatches. We observe a visual semantic internalization phenomenon in Vid-LLMs, indicating that critical visual semantics are implicitly encoded into text hidden states during deep-layer interactions, which renders raw visual inputs structurally redundant during deep inference. To address this, we propose the Sparrow framework, which first utilizes visually-aware text-anchored window attention via hidden state reuse to fully offload visual computation to the target model, and leverages intermediate-layer visual state bridging to train the draft model with semantic-rich intermediate states, thereby filtering out low-level visual noise. Additionally, a multi-token prediction strategy is introduced to bridge the training-inference distribution shift. Experiments show that Sparrow achieves an average speedup of 2.82x even with 25k visual tokens, effectively resolving the performance degradation in long sequences and offering a practical solution for real-time long video tasks. 

---
