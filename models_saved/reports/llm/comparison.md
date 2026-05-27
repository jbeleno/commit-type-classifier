# LLM sweep — commit message generation

Five locally-hosted LLMs × four prompting strategies × **n = 50** stratified
test commits. Inference temperature 0.2, top_p 0.9, seed 42 throughout.
Same Apple-Silicon Mac (16 GB unified memory), one model loaded at a time.

## Full leaderboard

| model                            | strategy         | type-match   | in-target   | classif. agree   | parse fail   |   bleu |   rouge_L |   p50 ms |   tokens |
|:---------------------------------|:-----------------|:-------------|:------------|:-----------------|:-------------|-------:|----------:|---------:|---------:|
| phi3.5:3.8b-mini-instruct-q4_K_M | few_shot         | 36.0%        | 100.0%      | 42.0%            | 0.0%         |   7.18 |     0.192 |     5793 |    141.2 |
| llama3.2:3b-instruct-q4_K_M      | chain_of_thought | 32.0%        | 74.0%       | 40.0%            | 26.0%        |   0.63 |     0.143 |     3969 |    167.2 |
| qwen2.5-coder:3b                 | chain_of_thought | 30.0%        | 92.0%       | 42.0%            | 8.0%         |   1.76 |     0.194 |     3589 |    159.5 |
| phi3.5:3.8b-mini-instruct-q4_K_M | zero_shot        | 30.0%        | 100.0%      | 46.0%            | 0.0%         |   4.15 |     0.179 |      964 |     15.7 |
| qwen2.5-coder:3b                 | json_mode        | 28.0%        | 100.0%      | 42.0%            | 0.0%         |   3.98 |     0.196 |     1313 |     32.7 |
| qwen2.5-coder:3b                 | few_shot         | 26.0%        | 100.0%      | 30.0%            | 0.0%         |   3.83 |     0.064 |      613 |      8.7 |
| phi3.5:3.8b-mini-instruct-q4_K_M | json_mode        | 26.0%        | 100.0%      | 48.0%            | 0.0%         |   4.1  |     0.199 |     1739 |     40.1 |
| qwen2.5-coder:1.5b               | chain_of_thought | 26.0%        | 84.0%       | 38.0%            | 16.0%        |   1.24 |     0.165 |     1990 |    158.7 |
| llama3.2:3b-instruct-q4_K_M      | json_mode        | 24.0%        | 100.0%      | 26.0%            | 0.0%         |   4.35 |     0.167 |     1212 |     27.7 |
| qwen2.5-coder:1.5b               | json_mode        | 20.0%        | 100.0%      | 42.0%            | 0.0%         |   0.68 |     0.18  |      825 |     30   |
| qwen2.5-coder:3b                 | zero_shot        | 20.0%        | 100.0%      | 48.0%            | 0.0%         |   1.76 |     0.203 |      701 |     15.2 |
| llama3.2:3b-instruct-q4_K_M      | zero_shot        | 20.0%        | 100.0%      | 38.0%            | 0.0%         |   1.7  |     0.167 |      640 |     13.2 |
| qwen2.5-coder:1.5b               | zero_shot        | 20.0%        | 100.0%      | 34.0%            | 0.0%         |   1.92 |     0.142 |      353 |     13   |
| qwen2.5-coder:1.5b               | few_shot         | 18.0%        | 100.0%      | 14.0%            | 0.0%         |   0.14 |     0.032 |      411 |     14   |
| deepseek-coder:1.3b              | json_mode        | 18.0%        | 86.0%       | 30.0%            | 14.0%        |   0.74 |     0.119 |      580 |     34.5 |
| phi3.5:3.8b-mini-instruct-q4_K_M | chain_of_thought | 2.0%         | 8.0%        | 36.0%            | 92.0%        |   0.26 |     0.086 |     4904 |    161.7 |
| deepseek-coder:1.3b              | few_shot         | 0.0%         | 0.0%        | 22.0%            | 100.0%       |   0.14 |     0.032 |     2319 |    158.9 |
| llama3.2:3b-instruct-q4_K_M      | few_shot         | 0.0%         | 12.0%       | 18.0%            | 88.0%        |   0.18 |     0.028 |     2177 |     81.8 |
| deepseek-coder:1.3b              | chain_of_thought | 0.0%         | 0.0%        | 28.0%            | 100.0%       |   0.07 |     0.053 |     1984 |    191.4 |
| deepseek-coder:1.3b              | zero_shot        | 0.0%         | 0.0%        | 22.0%            | 100.0%       |   0.2  |     0.061 |     1538 |    139.2 |

## Best strategy per model

| model                            | strategy         | type-match   |   bleu |   rouge_L |   p50 ms |
|:---------------------------------|:-----------------|:-------------|-------:|----------:|---------:|
| phi3.5:3.8b-mini-instruct-q4_K_M | few_shot         | 36.0%        |   7.18 |     0.192 |     5793 |
| llama3.2:3b-instruct-q4_K_M      | chain_of_thought | 32.0%        |   0.63 |     0.143 |     3969 |
| qwen2.5-coder:3b                 | chain_of_thought | 30.0%        |   1.76 |     0.194 |     3589 |
| qwen2.5-coder:1.5b               | chain_of_thought | 26.0%        |   1.24 |     0.165 |     1990 |
| deepseek-coder:1.3b              | json_mode        | 18.0%        |   0.74 |     0.119 |      580 |

## Findings

* **Best overall:** `phi3.5:3.8b-mini-instruct-q4_K_M` with `few_shot` reaches
  **36%** type-exact-match, BLEU
  7.18, ROUGE-L 0.192, p50 latency
  5793 ms.
* **Fastest workable model:** `qwen2.5-coder:1.5b` with `zero_shot` —
  p50 latency **352 ms**, type-exact-match
  20%. Useful as the default in interactive UIs.
* **Highest classifier-agreement:** `qwen2.5-coder:3b` with
  `zero_shot` — **48%** of generated
  messages get the correct type back when fed through the TF-IDF baseline.
  This is the most reliable input to the hybrid verifier pipeline.
* **Models that ignore the format contract** (parse-failure rate > 50 %):

| model                            | strategy         | parse_failure_rate   |
|:---------------------------------|:-----------------|:---------------------|
| phi3.5:3.8b-mini-instruct-q4_K_M | chain_of_thought | 92%                  |
| deepseek-coder:1.3b              | few_shot         | 100%                 |
| llama3.2:3b-instruct-q4_K_M      | few_shot         | 88%                  |
| deepseek-coder:1.3b              | chain_of_thought | 100%                 |
| deepseek-coder:1.3b              | zero_shot        | 100%                 |

  The pattern is consistent: base / code-completion checkpoints
  (`deepseek-coder:1.3b`) and few-shot or chain-of-thought modes on
  smaller instruct models tend to drift away from the
  `<type>(<scope>): <subject>` schema. JSON-mode mitigates the problem
  because the runtime enforces the structure.

## Relationship to the discriminative track

The five classifiers reported in `models_saved/reports/comparison.md`
(test split, n = 5 845) reach **70.93 % accuracy / 0.66 macro-F1** with
the TF-IDF baseline — a strictly easier task (5-way classification given
the full message and diff). The generative task asks the model to
*author the message from the diff alone*, which is open-ended and
fundamentally harder; the 36% type-match
of the best LLM is therefore not directly comparable.

The classical baseline is **reused** in the hybrid pipeline
(`src/llm/hybrid.py`) as a fast verifier: when the LLM emits a type
that the baseline disagrees with at confidence ≥ 0.60, the verifier
wins. The discriminative work is preserved as infrastructure rather
than discarded.
