from __future__ import annotations

from src.pipeline.runner import build_parser, run_pipeline


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
