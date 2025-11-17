# Evaluating LLMs' Reasoning

In this repository, we aim to evaluate the reasoning capabilities of recent LLMs beyond outcome-based benchmarks. Since using an LLM directly to verify the correctness of an answer is not a reliable approach, our main assumption is that it is easier to generate incorrect propositions.
Therefore, we suggest having a verifier that challenges the reasoning chain of an LLM with wrong propositions, and once the LLM falls for them, it has failed the test.

This repository contains a toy demonstration of this approach, where we take AIME 2024 as a dataset, and instead of having a systematic verifier, we use pre-generated incorrect answers from an LLM prompted to produce them.
To avoid a hackable evaluation where the evaluated LLM will never agree with the wrong answers, we randomize the process and sometimes include correct answers to avoid systematic disagreement. The correct answers are taken from the AIME 2024 proposed solutions.

# Results

<div align="center">
  <img src="https://github.com/rbouftini/ReasoningEval/blob/main/imgs/evaluation.png?raw=true" alt="Fig" width="60%"/>
  <br/>
  <em>
    Benchmarking LLM reasoning with an outcome-based evaluation (correctness of the final result) vs. the debate-like approach presented.
  </em>
</div>

## Getting Started

1. **Setup Python Environment**

   ```bash
   bash eval.sh  
   ```

2. **Run Evaluation**

   For now, evaluation only supports OpenAI models available through the API.

   * **Export your OpenAI API key**
  
     
   ```bash
   echo "OPENAI_API_KEY=YOUR_OPENAI_API_KEY" > .env  
   ```

   * **Run Script**

   Execute `eval.py` to evaluate a supported model. For example:

   ```bash
   python eval.py --model=gpt-5.1  
   ```

   For additional information about arguments, you can run:

   ```bash
   python eval.py --help  
   ```

   ```
   usage: eval.py [-h] [--generate] --model MODEL

   Evaluate LLM performance

   options:
     -h, --help     show this help message and exit
     --generate     Generate wrong solutions
     --model MODEL  Model to evaluate
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
