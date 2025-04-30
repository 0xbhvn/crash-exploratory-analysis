#!/bin/bash

# Activate virtual environment
source crash_env/bin/activate

# Run the streak processor with WebSocket mode enabled
python api/streak_processor.py --websocket

# Note: Add --test flag for testing with a small number of games
# python api/streak_processor.py --websocket --test 