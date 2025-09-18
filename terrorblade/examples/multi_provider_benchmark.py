#!/usr/bin/env python3
"""
Multi-Provider LLM Benchmark for Terrorblade

Tests multiple LLM providers (OpenAI, OpenRouter, DeepInfra) with time and token tracking.
Queries a random cluster from analyze_dialogues and compares responses.

Usage:
    python multi_provider_benchmark.py
    python multi_provider_benchmark.py --test-mode
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.examples.analyze_dialogues import SearchParams, SentimentAnalyser
from terrorblade.examples.prompts.promptinator import Promptinator
from terrorblade.utils.config import get_db_path

load_dotenv()


@dataclass
class BenchmarkResult:
    """Results from a single provider benchmark."""
    provider: str
    model: str
    response_content: str
    tokens_used: int | None
    cost: float | None
    response_time: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class MultiProviderBenchmark:
    """Benchmark multiple LLM providers on dialogue analysis."""

    def __init__(self, phone: str = "+79992004210", db_path: str = "auto"):
        self.console = Console()
        self.phone = phone.replace("+", "")
        self.db = TelegramDatabase(db_path=get_db_path(db_path), read_only=True)

        # Initialize analyzer for getting clusters
        self.analyzer = SentimentAnalyser(phone, db_path)

        # Provider configurations with specific models
        self.provider_configs = {
            "openai": {
                "models": ["gpt-4o-mini"],
                "env_key": "OPENAI_API_KEY"
            },
            "openrouter": {
                "models": ["anthropic/claude-3.5-sonnet"],
                "env_key": "OPENROUTER_API_KEY"
            },
            "deepinfra": {
                "models": ["meta-llama/Meta-Llama-3.1-8B-Instruct"],
                "env_key": "DEEPINFRA_API_KEY"
            }
        }

    def check_api_keys(self) -> dict[str, bool]:
        """Check which API keys are available."""
        available = {}
        for provider, config in self.provider_configs.items():
            api_key = os.getenv(config["env_key"])
            available[provider] = api_key is not None
        return available

    def get_random_cluster(self, test_mode: bool = False) -> tuple[dict[str, Any], list[str]] | None:
        """Get a random cluster from analyze_dialogues."""
        try:
            if test_mode:
                # Return simple test data
                return {
                    "chat_name": "Test Chat",
                    "message_count": 1,
                    "total_words": 4,
                    "participants": 1,
                    "avg_words_per_message": 4.0
                }, ["test: what's 0 +- 1?"]

            # Get clusters using analyzer
            _, quantiles = self.analyzer.analyze_word_quantiles()

            params = SearchParams(
                min_words=max(int(quantiles.get("q99.7", 50)), 50),
                min_consecutive=5,
                time_window_hours=1,
                overlap=10,
                chat_id=None
            )

            groups_df, all_messages_df = self.analyzer.find_long_message_groups(
                min_words=params.min_words,
                min_consecutive=params.min_consecutive,
                time_window_hours=params.time_window_hours,
                overlap=params.overlap,
                chat_id=params.chat_id
            )

            if groups_df.is_empty():
                self.console.print("❌ No clusters found", style="red")
                return None

            # Select first cluster (or random if you prefer)
            group_row = groups_df.sort("total_words", descending=True).row(0, named=True)
            group_id = group_row["group_id"]
            group_messages = all_messages_df.filter(pl.col("group_id") == group_id)

            # Format messages
            messages_text = []
            for row in group_messages.iter_rows(named=True):
                timestamp = (
                    row["date"].strftime("%H:%M")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"])
                )
                messages_text.append(
                    f"[{timestamp}] ({row['word_count']}w): {row['text']}"
                )

            group_data = {
                "chat_name": group_row["chat_name"],
                "message_count": group_row["message_count"],
                "total_words": group_row["total_words"],
                "participants": group_row["participants"],
                "avg_words_per_message": group_row["avg_words_per_message"]
            }

            return group_data, messages_text

        except Exception as e:
            self.console.print(f"Error getting cluster: {e}", style="red")
            return None

    def benchmark_provider(
        self,
        provider: str,
        model: str,
        query: str,
        prompt_file: str = "prompt_1.md"
    ) -> BenchmarkResult:
        """Benchmark a single provider."""
        try:
            # Initialize provider
            promptinator = Promptinator(provider=provider, model=model)

            # Time the request
            start_time = time.time()

            # Make the request
            if query == "test: what's 0 +- 1?":
                # Simple test query
                response = promptinator.query(
                    user_input=query,
                    temperature=0.7
                )
            else:
                # Use prompt file for real analysis
                response = promptinator.query(
                    user_input=query,
                    prompt_file=prompt_file,
                    temperature=0.7
                )

            end_time = time.time()
            response_time = end_time - start_time

            return BenchmarkResult(
                provider=provider,
                model=model,
                response_content=response.content,
                tokens_used=response.tokens_used,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost=response.cost,
                response_time=response_time,
                error=response.error
            )

        except Exception as e:
            return BenchmarkResult(
                provider=provider,
                model=model,
                response_content="",
                tokens_used=None,
                input_tokens=None,
                output_tokens=None,
                cost=None,
                response_time=0.0,
                error=str(e)
            )

    def run_benchmark(self, test_mode: bool = False, prompt_file: str = "prompt_1.md") -> list[BenchmarkResult]:
        """Run benchmark across all available providers concurrently."""
        self.console.print("🚀 Starting Multi-Provider LLM Benchmark (Concurrent)", style="bold cyan")

        # Check API keys
        available_providers = self.check_api_keys()

        if not any(available_providers.values()):
            self.console.print("❌ No API keys found. Please set environment variables:", style="red")
            for config in self.provider_configs.values():
                self.console.print(f"  export {config['env_key']}=your_key_here")
            return []

        # Get cluster data
        cluster_data = self.get_random_cluster(test_mode)
        if not cluster_data:
            return []

        group_data, messages_text = cluster_data

        # Format query
        query = messages_text[0] if test_mode else "\n".join(messages_text)

        # Display full cluster data from database
        if not test_mode:
            try:
                # Get the group_id to fetch the complete cluster
                group_row = self.analyzer.find_long_message_groups()[0].sort("total_words", descending=True).row(0, named=True)
                chat_id = group_row["chat_id"]
                
                cluster_sql = f"""
                SELECT *
                FROM {self.analyzer.messages_table} 
                WHERE chat_id = {chat_id}
                ORDER BY date
                LIMIT 100
                """
                cluster_df = pl.from_arrow(self.analyzer.db.db.execute(cluster_sql).arrow())
                
                self.console.print(f"\n📊 Full Cluster Database Dump:", style="bold blue")
                with pl.Config(tbl_rows=20, tbl_hide_column_data_types=True):
                    self.console.print(cluster_df)
            except Exception as e:
                self.console.print(f"Error fetching cluster data: {e}", style="red")

        if test_mode:
            self.console.print(f"\n🧪 Test Query: {query}", style="dim yellow")
        else:
            self.console.print(f"\n📝 Query Preview: {query[:100]}...", style="dim yellow")

        # Prepare tasks for concurrent execution
        tasks = []

        for provider, available in available_providers.items():
            if not available:
                continue

            config = self.provider_configs[provider]

            # Add each model for this provider to tasks
            for model in config["models"]:
                tasks.append((provider, model, query, prompt_file))

        if not tasks:
            self.console.print("❌ No providers available for testing", style="red")
            return []

        self.console.print(f"\n⚡ Running {len(tasks)} tests concurrently...", style="bold yellow")

        # Start timing for overall benchmark
        overall_start = time.time()

        # Run benchmarks concurrently using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.benchmark_provider, provider, model, query, prompt_file): (provider, model)
                for provider, model, query, prompt_file in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                provider, model = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.error:
                        self.console.print(f"❌ {provider.upper()} ({model}): {result.error}", style="red")
                    else:
                        self.console.print(f"✅ {provider.upper()} ({model}): {result.response_time:.2f}s", style="green")
                except Exception as e:
                    self.console.print(f"❌ {provider.upper()} ({model}): Exception - {e}", style="red")
                    results.append(BenchmarkResult(
                        provider=provider,
                        model=model,
                        response_content="",
                        tokens_used=None,
                        input_tokens=None,
                        output_tokens=None,
                        cost=None,
                        response_time=0.0,
                        error=str(e)
                    ))

        overall_time = time.time() - overall_start
        self.console.print(f"\n⚡ Concurrent execution completed in {overall_time:.2f}s", style="bold green")

        return results

    def display_results(self, results: list[BenchmarkResult]) -> None:
        """Display benchmark results in a formatted table."""
        if not results:
            self.console.print("❌ No results to display", style="red")
            return

        # Create summary table
        table = Table(title="🏆 Multi-Provider Benchmark Results", show_header=True)
        table.add_column("Provider", style="cyan", width=12)
        table.add_column("Model", style="blue", width=25)
        table.add_column("Time (s)", justify="right", style="yellow", width=10)
        table.add_column("Tokens (in/out)", justify="right", style="green", width=15)
        table.add_column("Cost ($)", justify="right", style="magenta", width=10)
        table.add_column("Status", style="white", width=15)

        successful_results = []
        failed_results = []

        for result in results:
            if result.error:
                status = f"❌ {result.error[:10]}..."
                failed_results.append(result)
            else:
                status = "✅ Success"
                successful_results.append(result)

            # Format tokens with input/output breakdown
            if result.input_tokens and result.output_tokens:
                tokens_str = f"{result.input_tokens:,}/{result.output_tokens:,}"
            elif result.tokens_used:
                tokens_str = f"{result.tokens_used:,}"
            else:
                tokens_str = "N/A"

            cost_str = f"{result.cost:.4f}" if result.cost else "N/A"

            table.add_row(
                result.provider.upper(),
                result.model,
                f"{result.response_time:.2f}",
                tokens_str,
                cost_str,
                status
            )

        self.console.print("\n")
        self.console.print(table)

        # Show response content for successful results
        for i, result in enumerate(successful_results):
            if not result.error:
                title = f"Response {i+1}: {result.provider.upper()} ({result.model})"
                content = result.response_content[:500] + ("..." if len(result.response_content) > 500 else "")
                self.console.print(Panel(content, title=title, border_style="blue"))

        # Calculate totals
        if successful_results:
            total_tokens = sum(r.tokens_used for r in successful_results if r.tokens_used)
            total_cost = sum(r.cost for r in successful_results if r.cost)

            self.console.print("\n📈 Summary:", style="bold cyan")
            self.console.print(f"  Successful requests: {len(successful_results)}/{len(results)}")
            self.console.print(f"  Total tokens: {total_tokens:,}")
            self.console.print(f"  Total cost: ${total_cost:.4f}")

            if successful_results:
                fastest = min(successful_results, key=lambda x: x.response_time)
                self.console.print(f"  Fastest: {fastest.provider.upper()} ({fastest.model}) - {fastest.response_time:.2f}s")

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
        self.analyzer.db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Provider LLM Benchmark")
    parser.add_argument("--test-mode", action="store_true",
                       help="Use simple test query instead of real cluster")
    parser.add_argument("--phone", default="+79992004210",
                       help="Phone number for database access")
    parser.add_argument("--prompt", default="prompt_1.md",
                       help="Prompt file to use")

    args = parser.parse_args()

    benchmark = MultiProviderBenchmark(phone=args.phone)

    try:
        results = benchmark.run_benchmark(test_mode=args.test_mode, prompt_file=args.prompt)
        benchmark.display_results(results)

    except KeyboardInterrupt:
        benchmark.console.print("\n👋 Benchmark cancelled!", style="yellow")
    except Exception as e:
        benchmark.console.print(f"\n❌ Benchmark failed: {e}", style="red")
    finally:
        benchmark.close()


if __name__ == "__main__":
    main()
