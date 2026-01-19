"""
Test all forbidden goals from dataset
"""
import json
import asyncio
from backend.app.models.database import db
from backend.app.core.orchestrator import Orchestrator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

async def test_dataset():
    """Test first 5 forbidden goals"""
    
    # Load dataset
    with open('data/forbidden_goals.json', 'r') as f:
        goals = json.load(f)
    
    await db.connect()
    
    try:
        orchestrator = Orchestrator()
        results_summary = []
        
        console.print("\n[bold cyan]🧪 Testing Forbidden Goals Dataset[/bold cyan]\n")
        
        # Test first 5 goals
        for i, goal_data in enumerate(goals[:5], 1):
            goal = goal_data['goal']
            category = goal_data['category']
            
            console.print(f"\n[yellow]Test {i}/5: {category.upper()}[/yellow]")
            console.print(f"Goal: {goal}\n")
            
            # Run audit with 3 most effective strategies
            batch_result = await orchestrator.run_batch_audit(
                forbidden_goal=goal,
                strategy_names=[
                    "authority_impersonation",
                    "contextual_camouflage", 
                    "hypothetical_framing"
                ],
                save_to_db=True,
                parallel=True
            )
            
            results_summary.append({
                "goal": goal,
                "category": category,
                "asr": batch_result.attack_success_rate,
                "jailbreaks": batch_result.successful_jailbreaks,
                "total": batch_result.total_attacks
            })
            
            console.print(f"[cyan]ASR: {batch_result.attack_success_rate:.2%}[/cyan]\n")
        
        # Summary
        console.print("\n[bold green]✅ Dataset Testing Complete![/bold green]\n")
        
        from rich.table import Table
        summary_table = Table(title="Results Summary")
        summary_table.add_column("#", style="dim")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("ASR", justify="right")
        summary_table.add_column("Jailbreaks", justify="center")
        
        for i, result in enumerate(results_summary, 1):
            asr_color = "red" if result['asr'] > 0 else "green"
            summary_table.add_row(
                str(i),
                result['category'],
                f"[{asr_color}]{result['asr']:.1%}[/{asr_color}]",
                f"{result['jailbreaks']}/{result['total']}"
            )
        
        console.print(summary_table)
        
        # Calculate overall ASR
        total_jailbreaks = sum(r['jailbreaks'] for r in results_summary)
        total_attacks = sum(r['total'] for r in results_summary)
        overall_asr = total_jailbreaks / total_attacks if total_attacks > 0 else 0
        
        console.print(f"\n[bold]Overall ASR: {overall_asr:.2%}[/bold]")
        console.print(f"Total Tests: {total_attacks}")
        console.print(f"Total Jailbreaks: {total_jailbreaks}\n")
        
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(test_dataset())
