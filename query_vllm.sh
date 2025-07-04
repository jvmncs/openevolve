#!/bin/bash

# Query OpenAI-compatible vLLM server
curl -X POST "https://modal-labs-jason-dev--openevolve-inference-inference-serve.modal.run/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat openevolve-vllm-secret.secret)" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 100
  }'
