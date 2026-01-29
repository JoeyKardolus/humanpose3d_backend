"""Service for executing the pipeline subprocess."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, TextIO

from src.application.dto.pipeline_execution_result import PipelineExecutionResult


class PipelineRunner:
    """Run the CLI pipeline and stream output."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def run(
        self,
        command: list[str],
        on_line: Callable[[str], None] | None = None,
    ) -> PipelineExecutionResult:
        """Execute the command and return collected stdout/stderr."""
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _stream_reader(stream: TextIO, sink: TextIO, collector: list[str]) -> None:
            for line in iter(stream.readline, ""):
                collector.append(line)
                if on_line:
                    on_line(line)
                sink.write(line)
                sink.flush()

        with subprocess.Popen(
            command,
            cwd=str(self._repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as process:
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Failed to attach to pipeline output streams.")

            stdout_thread = threading.Thread(
                target=_stream_reader,
                args=(process.stdout, sys.stdout, stdout_lines),
            )
            stderr_thread = threading.Thread(
                target=_stream_reader,
                args=(process.stderr, sys.stderr, stderr_lines),
            )
            stdout_thread.start()
            stderr_thread.start()
            return_code = process.wait()
            stdout_thread.join()
            stderr_thread.join()

        return PipelineExecutionResult(
            return_code=return_code,
            stdout_text="".join(stdout_lines),
            stderr_text="".join(stderr_lines),
        )
