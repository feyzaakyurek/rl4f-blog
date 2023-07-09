![image](new_method_image.jpg)

## Abstract

Despite their unprecedented success, even the largest language models make mistakes. Similar to how humans learn and improve using feedback, previous work proposed providing language models with natural language feedback to guide them in repairing their outputs. Because human-generated critiques are expensive to obtain, researchers have devised learned critique generators in lieu of human critics while assuming one can train downstream models to utilize generated feedback. However, this approach does not apply to black-box or limited access models such as ChatGPT, as they cannot be fine-tuned. Moreover, in the era of large general-purpose language agents, fine-tuning is neither computationally nor spatially efficient as it results in multiple copies of the network. In this work, we introduce RL4F (Reinforcement Learning for Feedback), a multi-agent collaborative framework where the critique generator is trained to maximize end-task performance of GPT-3, a fixed model more than 200 times its size. RL4F produces critiques that help GPT-3 revise its outputs. We study three datasets for action planning, summarization and alphabetization and show improvements (~5% on average) in multiple text similarity metrics over strong baselines across all three tasks.

*Afra Feyza Akyürek, Ekin Akyürek, Aman Madaan, Ashwin Kalyan, Peter Clark, Derry Wijaya, Niket Tandon*, ACL 2023

Link to paper: https://arxiv.org/abs/2305.08844

Code: https://github.com/feyzaakyurek/rl4f

## Motivation

Large language models continue to make mistakes. While they understand human-written critiques of their outputs, feedback from humans is not scalable to online settings. Hence, we propose to train a small language model conditioned to critique an LLM. We train a small Critique Model (220x times smaller than GPT-3.5) to critique GPT-3.5 (Task Model). We use task supervision (e.g. for summarization we use context and summary pairs) to train the Critique Model via reinforcement learning. Once we sample a critique utterance from the Critique Model, we use few-shot prompting to get a revised and improved answer from the Task Model.

## Highlights

RL4F provides a way to bootstrap LLM performance on a task without needing to train the large model using a small computational budget. It also caters to the situations where the task model is black-box which is the case for the major proprietary language models.


## Procedure

### How to use natural language critiques in model outputs?
We use few-shot prompt when we first sample an initial prediction and when we prompt the model to revise its the answer using the critique.


### RL4F: Training the critique model
Having sampled a critique utterance from the critique model we prompt the task model for a revision. We compare the revision to a ground truth output (e.g. human-written summary for the summarization task). We use automatic metrics to quantify the quality of the revision and use that as a reward signal.


### Applying RL4F iteratively
At evaluation, we sample critiques from the critique model iteratively and observe improvements for alphabetization task.


### Scaling RL4F
Our default critique model is a fine-tuned T5-large checkpoint by default. We conduct experiments usind differently-sized T5 models and find that increasing the model size increase the end-task performance.


### Results
RL4F yields improvements over retrieval-based, self-refinement and supervised baselines. Check out the paper for descriptions of these baselines and the appendix for a comparison to subsequent works such as [Self-Refine](https://arxiv.org/abs/2303.17651).


### Sample critiques from RL4F
Below are some examples where RL4F critiques were useful. There are also examples when the revised answer is not better than the initial answer even though the critiques were reasonable.

```
@article{akyurek2023rl4f,
  title={RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs},
  author={Aky{\"u}rek, Afra Feyza and Aky{\"u}rek, Ekin and Madaan, Aman and Kalyan, Ashwin and Clark, Peter and Wijaya, Derry and Tandon, Niket},
  journal={arXiv preprint arXiv:2305.08844},
  year={2023}
}
```
