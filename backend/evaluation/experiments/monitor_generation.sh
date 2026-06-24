#!/bin/bash
# Monitor script to track Ollama/dataset generation progress

echo "⏱️  Monitoring Ollama progress..."
echo ""

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Checking status..."
    
    # Check if Ollama is running
    if pgrep -q "llama-server"; then
        echo "  ✅ Ollama LLM server: RUNNING"
    else
        echo "  ❌ Ollama LLM server: STOPPED"
    fi
    
    # Check if dataset JSON exists
    if [ -f "/Users/forgetquentin/Desktop/CV-Agent/backend/evaluation/datasets/agent_test_suite_100.json" ]; then
        LINES=$(wc -l < "/Users/forgetquentin/Desktop/CV-Agent/backend/evaluation/datasets/agent_test_suite_100.json")
        SIZE=$(du -h "/Users/forgetquentin/Desktop/CV-Agent/backend/evaluation/datasets/agent_test_suite_100.json" | cut -f1)
        echo "  ✅ Dataset JSON: EXISTS ($LINES lines, $SIZE)"
        echo ""
        echo "🎉 Dataset generation COMPLETE!"
        break
    else
        echo "  ⏳ Dataset JSON: Generating... (checking Ollama processes)"
        ps aux | grep -i "ollama\|mistral" | grep -v grep | head -2 | sed 's/^/     /'
    fi
    
    echo ""
    sleep 30
done
