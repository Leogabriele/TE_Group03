"""
Command-Line Interface for LLM Security Auditor
"""

from unittest import result
import click
import asyncio
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.config import settings
from backend.app.models.database import db
from backend.app.core.orchestrator import Orchestrator
from backend.app.agents.strategies import list_all_strategies

console = Console()


@click.group()
def cli():
    """LLM Security Auditor - Command Line Interface"""
    pass


@cli.command()
@click.option('--goal', '-g', required=True, help='Forbidden goal to test')
@click.option('--strategy', '-s', required=True, help='Attack strategy name')
@click.option('--target', '-t', default=None, help='Target model (optional)')
@click.option('--provider', '-p', default=None, help='Target provider (groq/nvidia)')
@click.option('--save/--no-save', default=True, help='Save to database')
def test_single(goal, strategy, target, provider, save):
    """Test a single attack with one strategy"""
    asyncio.run(_test_single(goal, strategy, target, provider, save))


async def _test_single(goal, strategy, target, provider, save):
    """Async implementation of test_single"""
    
    console.print("\n[bold cyan]🛡️ LLM Security Auditor - Single Test[/bold cyan]\n")
    
    # Connect to database
    if save:
        with console.status("[bold green]Connecting to database..."):
            await db.connect()
    
    try:
        # Initialize orchestrator
        if target and provider:
            orchestrator = Orchestrator(target_provider=provider, target_model=target)
        else:
            orchestrator = Orchestrator()
        
        # Display configuration
        config_table = Table(title="Configuration", show_header=False)
        config_table.add_column("Key", style="cyan")
        config_table.add_column("Value", style="yellow")
        
        config_table.add_row("Forbidden Goal", goal[:50] + "..." if len(goal) > 50 else goal)
        config_table.add_row("Strategy", strategy)
        config_table.add_row("Target Model", orchestrator.target_model)
        config_table.add_row("Target Provider", orchestrator.target_provider)
        config_table.add_row("Save to DB", "Yes" if save else "No")
        
        console.print(config_table)
        console.print()
        
        # Run attack with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Running attack...", total=None)
            
            result = await orchestrator.run_single_attack(
                forbidden_goal=goal,
                strategy_name=strategy,
                save_to_db=save
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]✅ Attack Complete![/bold green]\n")
        
        # Results table
        results_table = Table(title="Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")
        
        results_table.add_row("Attack ID", result.attack.attack_id)
        verdict_text = result.evaluation.verdict.value
        verdict_color = "red" if result.success else "green"
        results_table.add_row("Verdict", f"[bold {verdict_color}]{verdict_text}[/bold {verdict_color}]")
        results_table.add_row("Confidence", f"{result.evaluation.confidence_score:.2%}")
        results_table.add_row("Execution Time", f"{result.total_time_ms}ms")
        results_table.add_row("Success", "🔴 JAILBROKEN" if result.success else "🟢 REFUSED")
        
        console.print(results_table)
        
        # Show generated prompt
        console.print("\n[bold cyan]Generated Attack Prompt:[/bold cyan]")
        console.print(Panel(result.attack.generated_prompt[:500] + "...", 
                           title="Prompt (truncated)", border_style="cyan"))
        
        # Show response
        console.print("\n[bold cyan]Target Model Response:[/bold cyan]")
        console.print(Panel(result.response.response_text[:500] + "...", 
                           title="Response (truncated)", border_style="cyan"))
        
        # Show reasoning
        console.print("\n[bold cyan]Judge Reasoning:[/bold cyan]")
        console.print(f"[yellow]{result.evaluation.reasoning}[/yellow]")
        
    finally:
        if save:
            await db.disconnect()


@cli.command()
@click.option('--goal', '-g', required=True, help='Forbidden goal to test')
@click.option('--all-strategies', is_flag=True, help='Use all available strategies')
@click.option('--strategies', '-s', multiple=True, help='Specific strategies to use')
@click.option('--target', '-t', default=None, help='Target model (optional)')
@click.option('--provider', '-p', default=None, help='Target provider (groq/nvidia)')
@click.option('--parallel/--sequential', default=True, help='Run in parallel or sequential')
@click.option('--save/--no-save', default=True, help='Save to database')
@click.option('--output', '-o', default=None, help='Output JSON file path')
def audit(goal, all_strategies, strategies, target, provider, parallel, save, output):
    """Run full audit with multiple strategies"""
    asyncio.run(_audit(goal, all_strategies, strategies, target, provider, parallel, save, output))


async def _audit(goal, all_strategies, strategies, target, provider, parallel, save, output):
    """Async implementation of audit"""
    
    console.print("\n[bold cyan]🛡️ LLM Security Auditor - Full Audit[/bold cyan]\n")
    
    # Determine strategies to use
    if all_strategies:
        strategy_list = [s.strategy_type.value for s in list_all_strategies()]
    elif strategies:
        strategy_list = list(strategies)
    else:
        console.print("[red]Error: Must specify either --all-strategies or --strategies[/red]")
        return
    
    # Connect to database
    if save:
        with console.status("[bold green]Connecting to database..."):
            await db.connect()
    
    try:
        # Initialize orchestrator
        if target and provider:
            orchestrator = Orchestrator(target_provider=provider, target_model=target)
        else:
            orchestrator = Orchestrator()
        
        # Display configuration
        console.print(f"[cyan]Forbidden Goal:[/cyan] {goal}")
        console.print(f"[cyan]Strategies:[/cyan] {len(strategy_list)}")
        console.print(f"[cyan]Target:[/cyan] {orchestrator.target_provider}/{orchestrator.target_model}")
        console.print(f"[cyan]Mode:[/cyan] {'Parallel' if parallel else 'Sequential'}")
        console.print()
        
        # Run batch audit with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Running {len(strategy_list)} attacks...", 
                total=None
            )
            
            batch_result = await orchestrator.run_batch_audit(
                forbidden_goal=goal,
                strategy_names=strategy_list,
                save_to_db=save,
                parallel=parallel
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]✅ Audit Complete![/bold green]\n")
        
        # Summary table - FIXED
        summary_table = Table(title="Audit Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value")
        
        # Format jailbreaks
        jailbreak_count = str(batch_result.successful_jailbreaks)
        if batch_result.successful_jailbreaks > 0:
            jailbreak_display = f"[bold red]{jailbreak_count}[/bold red]"
        else:
            jailbreak_display = f"[yellow]{jailbreak_count}[/yellow]"
        
        # Format ASR
        asr_value = batch_result.attack_success_rate
        if asr_value > 0:
            asr_display = f"[bold red]{asr_value:.2%}[/bold red]"
        else:
            asr_display = f"[bold green]{asr_value:.2%}[/bold green]"
        
        summary_table.add_row("Audit ID", f"[yellow]{batch_result.audit_id}[/yellow]")
        summary_table.add_row("Total Attacks", f"[yellow]{str(batch_result.total_attacks)}[/yellow]")
        summary_table.add_row("Successful Jailbreaks", jailbreak_display)
        summary_table.add_row("Attack Success Rate", asr_display)
        summary_table.add_row("Execution Time", f"[yellow]{batch_result.total_execution_time_ms/1000:.1f}s[/yellow]")
        
        console.print(summary_table)
        
        # Detailed results table - FIXED
        console.print("\n[bold cyan]Detailed Results:[/bold cyan]\n")
        
        detailed_table = Table(show_header=True)
        detailed_table.add_column("#", style="dim")
        detailed_table.add_column("Strategy", style="cyan")
        detailed_table.add_column("Verdict")
        detailed_table.add_column("Confidence", justify="right")
        detailed_table.add_column("Time", justify="right")
        
        for i, result in enumerate(batch_result.results, 1):
            verdict = result.evaluation.verdict.value
            
            if result.success:
                verdict_display = f"[red]{verdict}[/red]"
            else:
                verdict_display = f"[green]{verdict}[/green]"
            
            detailed_table.add_row(
                str(i),
                result.attack.strategy_name,
                verdict_display,
                f"[yellow]{result.evaluation.confidence_score:.2%}[/yellow]",
                f"[yellow]{result.total_time_ms}ms[/yellow]"
            )
        
        console.print(detailed_table)
        
        # Save to file if requested
        if output:
            output_data = {
                "audit_id": batch_result.audit_id,
                "forbidden_goal": batch_result.forbidden_goal,
                "target_model": batch_result.target_model,
                "total_attacks": batch_result.total_attacks,
                "successful_jailbreaks": batch_result.successful_jailbreaks,
                "attack_success_rate": batch_result.attack_success_rate,
                "results": [
                    {
                        "strategy": r.attack.strategy_name,
                        "verdict": r.evaluation.verdict.value,
                        "confidence": r.evaluation.confidence_score,
                        "time_ms": r.total_time_ms
                    }
                    for r in batch_result.results
                ]
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            console.print(f"\n[green]✅ Results saved to {output}[/green]")
        
    finally:
        if save:
            await db.disconnect()


@cli.command()
def list_strategies():
    """List all available attack strategies"""
    
    console.print("\n[bold cyan]📋 Available Attack Strategies[/bold cyan]\n")
    
    strategies = list_all_strategies()
    
    table = Table(show_header=True)
    table.add_column("#", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Difficulty", style="magenta")
    table.add_column("Description", style="white")
    
    for i, strategy in enumerate(strategies, 1):
        metadata = strategy.get_metadata()
        table.add_row(
            str(i),
            metadata['name'],
            metadata['type'],
            metadata['difficulty'],
            metadata['description'][:60] + "..."
        )
    
    console.print(table)
    console.print(f"\n[green]Total: {len(strategies)} strategies[/green]\n")


@cli.command()
def check_apis():
    """Check connectivity to all APIs"""
    asyncio.run(_check_apis())


async def _check_apis():
    """Async implementation of check_apis"""
    
    console.print("\n[bold cyan]🔍 Checking API Connectivity[/bold cyan]\n")
    
    results = {}
    
    # Check Groq
    with console.status("[bold green]Testing Groq API..."):
        try:
            from backend.app.core.llm_clients import GroqClient
            client = GroqClient(settings.GROQ_API_KEY, "llama3-8b-8192")
            response = await client.generate_async("Say 'test'", max_tokens=10)
            results['Groq'] = ('✅', 'Connected', 'green')
        except Exception as e:
            results['Groq'] = ('❌', f'Failed: {str(e)[:50]}', 'red')
    
    # Check NVIDIA
    with console.status("[bold green]Testing NVIDIA API..."):
        try:
            from backend.app.core.llm_clients import NVIDIAClient
            client = NVIDIAClient(settings.NVIDIA_API_KEY, "meta/llama3-70b-instruct")
            response = await client.generate_async("Say 'test'", max_tokens=10)
            results['NVIDIA'] = ('✅', 'Connected', 'green')
        except Exception as e:
            results['NVIDIA'] = ('❌', f'Failed: {str(e)[:50]}', 'red')
    
    # Check MongoDB
    with console.status("[bold green]Testing MongoDB..."):
        try:
            await db.connect()
            count = await db.get_collection_count("attacks")
            results['MongoDB'] = ('✅', f'Connected ({count} attacks)', 'green')
            await db.disconnect()
        except Exception as e:
            results['MongoDB'] = ('❌', f'Failed: {str(e)[:50]}', 'red')
    
    # Display results
    table = Table(title="API Status", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="white")
    
    for service, (icon, details, color) in results.items():
        table.add_row(service, f"[{color}]{icon}[/{color}]", details)
    
    console.print(table)
    
    # Overall status
    all_ok = all(r[0] == '✅' for r in results.values())
    if all_ok:
        console.print("\n[bold green]✅ All services are operational![/bold green]\n")
    else:
        console.print("\n[bold red]❌ Some services are not working. Check configuration.[/bold red]\n")


@cli.command()
@click.option('--days', '-d', default=30, help='Number of days to analyze')
def stats(days):
    """Show statistics from database"""
    asyncio.run(_stats(days))


@cli.command()
@click.option('--goal', '-g', required=True, help='Forbidden goal to test')
@click.option('--strategy', '-s', required=True, help='Initial attack strategy')
@click.option('--max-turns', '-t', default=5, help='Maximum conversation turns')
@click.option('--save/--no-save', default=True, help='Save to database')
def test_multiturn(goal, strategy, max_turns, save):
    """Test multi-turn adaptive attack"""
    asyncio.run(_test_multiturn(goal, strategy, max_turns, save))


async def _test_multiturn(goal, strategy, max_turns, save):
    """Async implementation of test_multiturn"""
    
    console.print("\n[bold cyan]🔄 LLM Security Auditor - Multi-Turn Test[/bold cyan]\n")
    
    if save:
        with console.status("[bold green]Connecting to database..."):
            await db.connect()
    
    try:
        orchestrator = Orchestrator()
        
        # Display config
        console.print(f"[cyan]Forbidden Goal:[/cyan] {goal}")
        console.print(f"[cyan]Initial Strategy:[/cyan] {strategy}")
        console.print(f"[cyan]Max Turns:[/cyan] {max_turns}")
        console.print(f"[cyan]Target:[/cyan] {orchestrator.target_model}\n")
        
        # Run multi-turn attack
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Running multi-turn attack...", total=None)
            
            result = await orchestrator.run_multi_turn_attack(
                forbidden_goal=goal,
                strategy_name=strategy,
                max_turns=max_turns,
                save_to_db=save
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]✅ Multi-Turn Attack Complete![/bold green]\n")
        
        # Summary table
        summary_table = Table(title="Attack Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value")
        
        jailbroken = result['jailbroken']
        verdict_color = "red" if jailbroken else "green"
        
        summary_table.add_row("Conversation ID", result['conversation_id'][:16] + "...")
        summary_table.add_row("Total Turns", str(result['total_turns']))
        summary_table.add_row("Final Verdict", f"[{verdict_color}]{result['final_verdict']}[/{verdict_color}]")
        summary_table.add_row("Jailbroken", f"[{verdict_color}]{'YES' if jailbroken else 'NO'}[/{verdict_color}]")
        
        if result['turns_to_jailbreak']:
            summary_table.add_row("Turns to Jailbreak", f"[red]{result['turns_to_jailbreak']}[/red]")
        
        console.print(summary_table)
        
        # Turn-by-turn results
        console.print("\n[bold cyan]Turn-by-Turn Results:[/bold cyan]\n")
        
        turns_table = Table(show_header=True)
        turns_table.add_column("Turn", style="dim")
        turns_table.add_column("Verdict")
        turns_table.add_column("Confidence", justify="right")
        turns_table.add_column("Attack Preview", style="white")
        
        for turn_data in result['turns']:
            verdict = turn_data['verdict']
            verdict_color = "red" if verdict == "JAILBROKEN" else "yellow" if verdict == "PARTIAL" else "green"
            
            turns_table.add_row(
                str(turn_data['turn']),
                f"[{verdict_color}]{verdict}[/{verdict_color}]",
                f"{turn_data['confidence']:.1%}",
                turn_data['prompt_preview']
            )
        
        console.print(turns_table)
        console.print()
        
    finally:
        if save:
            await db.disconnect()


async def _stats(days):
    """Async implementation of stats"""
    
    console.print(f"\n[bold cyan]📊 Statistics (Last {days} days)[/bold cyan]\n")
    
    await db.connect()
    
    try:
        # Get counts
        attacks_count = await db.get_collection_count("attacks")
        responses_count = await db.get_collection_count("responses")
        evaluations_count = await db.get_collection_count("evaluations")
        
        # Get ASR
        asr = await db.calculate_asr(days=days)
        
        # Display
        table = Table(title="Database Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Total Attacks", str(attacks_count))
        table.add_row("Total Responses", str(responses_count))
        table.add_row("Total Evaluations", str(evaluations_count))
        table.add_row("Attack Success Rate", f"{asr:.2%}")
        
        console.print(table)
        console.print()
        
    finally:
        await db.disconnect()


if __name__ == '__main__':
    cli()
