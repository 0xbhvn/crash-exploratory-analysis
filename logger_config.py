#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logger configuration for Crash Game 10× Streak Analysis.

This module sets up rich logging with progress bars for the application.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    ProgressColumn
)

# Create console for rich output
console = Console()

# Create a custom progress bar


def create_progress_bar():
    """Create a rich progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )


class RichProgressHandler(logging.Handler):
    """Logging handler that uses rich progress bars."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.progress = None
        self.task_id = None

    def emit(self, record):
        # Skip progress messages if no progress bar is set up
        if hasattr(record, 'progress') and self.progress is not None:
            if record.progress >= 0 and record.progress <= 1:
                self.progress.update(
                    self.task_id, completed=int(record.progress * 100))


def setup_logging(log_file: Optional[str] = 'logs/crash_analysis.log', level=logging.INFO):
    """
    Set up rich logging for the application.

    Args:
        log_file: Optional path to a log file (default: logs/crash_analysis.log)
        level: Logging level

    Returns:
        Configured logger
    """
    # Configure rich handler
    rich_handler = RichHandler(
        rich_tracebacks=True,
        console=console,
        enable_link_path=False
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure handlers
    handlers = [rich_handler]

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(message)s",
    )

    # Return the root logger
    return logging.getLogger()


# Create a progress logger
progress_handler = RichProgressHandler()
progress_logger = logging.getLogger('progress')
progress_logger.setLevel(logging.INFO)
progress_logger.addHandler(progress_handler)
progress_logger.propagate = False


def start_progress(description: str, total: int = 100):
    """
    Start a progress tracking task.

    Args:
        description: Description of the task
        total: Total steps for completion (default 100)
    """
    progress = create_progress_bar()
    task_id = progress.add_task(description, total=total)
    progress_handler.progress = progress
    progress_handler.task_id = task_id
    progress.start()
    return progress, task_id


def update_progress(progress_value: float):
    """
    Update progress of the current task.

    Args:
        progress_value: Progress value between 0.0 and 1.0
    """
    progress_logger.info("", extra={"progress": progress_value})


def complete_progress(progress):
    """
    Mark progress as complete and stop the progress bar.

    Args:
        progress: Progress bar object
    """
    if progress:
        progress.stop()

# Utility to print styled messages


def print_info(message: str):
    """Print an info message with styling."""
    console.print(f"[bold blue]INFO:[/] {message}")


def print_success(message: str):
    """Print a success message with styling."""
    console.print(f"[bold green]SUCCESS:[/] {message}")


def print_warning(message: str):
    """Print a warning message with styling."""
    console.print(f"[bold yellow]WARNING:[/] {message}")


def print_error(message: str):
    """Print an error message with styling."""
    console.print(f"[bold red]ERROR:[/] {message}")
