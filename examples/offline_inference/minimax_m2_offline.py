# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Offline inference script for MiniMax-M2.5 with tensor parallelism.

Usage:
    python minimax_m2_offline.py

Model: /mnt/data3/models/MiniMax/MiniMax-M2.5
"""

from vllm import LLM, SamplingParams

# Model path
MODEL_PATH = "/mnt/data3/models/MiniMax/MiniMax-M2.5"

# Default sampling parameters
DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)


def run_minimax_offline():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        trust_remote_code=True,
        attention_backend="TOKEN_SPARSE",
    )
    prompt_0 = "What is the capital of France?" * 100000
    prompt_1 = "Write a short poem about the sea." * 100000
    prompt_2 = "Explain quantum computing in simple terms." * 100000
    prompts = [
        prompt_0[:196600],
        prompt_1[:196600],
        prompt_2[:196600],
    ]

    sampling_params = DEFAULT_SAMPLING_PARAMS

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 60)


if __name__ == "__main__":
    run_minimax_offline()
