import argparse
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ds_star import DSStar
from src.config import DSStarConfig, LLMConfig


def print_result(code: str, result: str, use_rich: bool = True):
    """Print the result in a nice format."""
    if use_rich and RICH_AVAILABLE:
        console = Console()

        console.print("\n[bold cyan]Generated Code:[/bold cyan]")
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Solution Code", border_style="cyan"))

        console.print("\n[bold green]Result:[/bold green]")
        console.print(Panel(result, title="Output", border_style="green"))
    else:
        print("\n" + "=" * 80)
        print("GENERATED CODE:")
        print("=" * 80)
        print(code)
        print("\n" + "=" * 80)
        print("RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DS-STAR: Data Science Agent via Iterative Planning and Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "question",
        type=str,
        help="The question to answer using the data files",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data files (default: data)",
    )

    parser.add_argument(
        "--guidelines",
        type=str,
        help="Formatting guidelines for the output",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "anthropic"],
        help="LLM provider (default: gemini)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: provider-specific default)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or use environment variable)",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Maximum number of refinement rounds (default: 20)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich formatting",
    )

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.model:
        model_defaults = {
            "gemini": "gemini-2.5-pro",
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
        }
        args.model = model_defaults[args.provider]

    config = DSStarConfig(
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
        ),
        max_rounds=args.max_rounds,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        debug_mode=args.debug,
    )

    use_rich = RICH_AVAILABLE and not args.no_rich
    if use_rich:
        console = Console()
        console.print(f"\n[bold blue]DS-STAR Configuration:[/bold blue]")
        console.print(f"  Provider: [cyan]{args.provider}[/cyan]")
        console.print(f"  Model: [cyan]{args.model}[/cyan]")
        console.print(f"  Data Directory: [cyan]{args.data_dir}[/cyan]")
        console.print(f"  Max Rounds: [cyan]{args.max_rounds}[/cyan]")
        console.print(f"  Log Directory: [cyan]{args.log_dir}[/cyan]")
        console.print(f"\n[bold yellow]Question:[/bold yellow] {args.question}\n")
    else:
        print(f"\nDS-STAR Configuration:")
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model}")
        print(f"  Data Directory: {args.data_dir}")
        print(f"  Max Rounds: {args.max_rounds}")
        print(f"  Log Directory: {args.log_dir}")
        print(f"\nQuestion: {args.question}\n")

    try:
        agent = DSStar(config)

        final_code, final_result = agent.run(
            question=args.question,
            data_dir=args.data_dir,
            guidelines=args.guidelines,
        )

        print_result(final_code, final_result, use_rich)

        if use_rich:
            console = Console()
            console.print(f"\n[bold green]✓[/bold green] Logs saved to: [cyan]{args.log_dir}[/cyan]")
        else:
            print(f"\nLogs saved to: {args.log_dir}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.debug:
            raise
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

