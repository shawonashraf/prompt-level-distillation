#!/bin/bash
# Usage: ./smoke_test.sh [host:port] [model]
# Defaults: teacher on localhost:8000. For the student: ./smoke_test.sh localhost:8001 google/gemma-4-E2B-it
HOST="${1:-localhost:8000}"
MODEL="${2:-google/gemma-4-26B-A4B-it}"

curl -sf "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: OK\"}],
        \"max_tokens\": 10,
        \"temperature\": 0
    }" | python3 -m json.tool
