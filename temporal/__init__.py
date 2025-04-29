#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal Analysis Package for Crash Game 10× Streak Analysis.

This package implements enhanced time-series modeling with:
1. Strict temporal separation to eliminate data leakage
2. Purely historical features (no current streak properties)
3. Focus on genuinely predictive temporal patterns
4. Realistic evaluation metrics for time-series prediction
5. Transition analysis to measure pattern prediction power
"""

from temporal.loader import load_data
from temporal.features import create_temporal_features
from temporal.splitting import temporal_train_test_split
from temporal.training import train_temporal_model
from temporal.evaluation import analyze_temporal_performance, analyze_recall_improvements
from temporal.prediction import make_temporal_prediction
from temporal.app import parse_arguments, main
