#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate the model bundle and check percentile values.
"""

import joblib
import numpy as np
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_and_validate_model(model_path="output/temporal_model.pkl"):
    """Load model bundle and validate contents."""
    try:
        console.print(Panel(
            f"Loading model from {model_path}", title="Model Validation", style="blue"))

        if not os.path.exists(model_path):
            console.print(
                f"[red]ERROR: Model file not found at {model_path}[/red]")
            return None

        with open(model_path, "rb") as f:
            bundle = joblib.load(f)

        # Verify bundle contents
        console.print("[green]Model bundle loaded successfully[/green]")

        # Check keys
        keys = bundle.keys()
        console.print(
            f"[cyan]Bundle contains {len(keys)} keys: {', '.join(keys)}[/cyan]")

        # Check percentile values
        if "percentile_values" in bundle:
            p_values = bundle["percentile_values"]
            console.print(f"[green]Percentile values: {p_values}[/green]")
        else:
            console.print(
                "[yellow]WARNING: No percentile_values found in model bundle[/yellow]")

        # Check features
        if "feature_cols" in bundle:
            features = bundle["feature_cols"]
            console.print(f"[cyan]Model uses {len(features)} features[/cyan]")
            for i, feat in enumerate(features[:10]):  # Show first 10 features
                console.print(f"  [blue]{i+1}.[/blue] {feat}")
            if len(features) > 10:
                console.print(f"  [blue]...and {len(features)-10} more[/blue]")

        # Check scaler
        if "scaler" in bundle:
            scaler = bundle["scaler"]
            console.print("[green]Scaler found in bundle[/green]")

            # Test scaler on random data
            try:
                X = np.random.rand(5, len(bundle.get("feature_cols", [10])))
                scaled_x = scaler.transform(X)
                console.print(
                    f"[green]Scaler test successful. Mean after scaling: {scaled_x.mean():.4f}[/green]")
            except Exception as e:
                console.print(f"[red]Scaler test failed: {str(e)}[/red]")
        else:
            console.print(
                "[yellow]WARNING: No scaler found in model bundle[/yellow]")

        return bundle

    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}[/red]")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate model bundle")
    parser.add_argument("--model", default="output/temporal_model.pkl",
                        help="Path to model pickle file")
    args = parser.parse_args()

    bundle = load_and_validate_model(args.model)

    if bundle:
        console.print(Panel("Model validation complete", style="green"))
    else:
        console.print(Panel("Model validation failed", style="red"))
