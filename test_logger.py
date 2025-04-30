#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for websocket_logger functionality"""

import asyncio
import json
from utils.websocket_logger import log_game, log_streak_processing, log_websocket_connection
from utils.logger_config import console


async def test_websocket_logger():
    print("Starting websocket logger test...")
    console.print("[bold]Testing rich console output[/bold]")

    # Test connection
    print("Testing connection log...")
    log_websocket_connection(
        'connected', 'wss://crashed-proxy-production.up.railway.app/ws')

    # Test typical game data
    print("Testing 19.44x game log...")
    game_data = {
        'gameId': '8086354',
        'crashPoint': 19.44,
        'hashValue': 'e8ca30e84b84970d133b70f2d1ee2681a16a44d23647074e78c8a6ffdbba6d62',
        'crashedFloor': 19,
        'endTime': '2025-04-30T08:24:45.858000+05:30'
    }
    log_game(game_data)

    # Test regular game
    print("Testing 2.48x game log...")
    game_data = {
        'gameId': '8086355',
        'crashPoint': 2.48,
        'hashValue': '0984ab3599fc1b9d8974b63768c521eeac0c2cb288237393285d77e9d28457af',
        'crashedFloor': 2,
        'endTime': '2025-04-30T08:25:12.415000+05:30'
    }
    log_game(game_data)

    # Test near-10x game
    print("Testing 9.65x game log...")
    game_data = {
        'gameId': '8086356',
        'crashPoint': 9.65,
        'hashValue': '20bb293c3303ce439caed283721ac06f8b9fe2f39618f28f67f3755d263b0542',
        'crashedFloor': 9,
        'endTime': '2025-04-30T08:26:01.821000+05:30'
    }
    log_game(game_data)

    # Test streak processing
    print("Testing streak processing log...")
    log_streak_processing(111446, '8086354-8086354')

    print("Test completed.")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_websocket_logger())
