"""
Test API connectivity script
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.config import settings
from backend.app.core.llm_clients import GroqClient, NVIDIAClient
from backend.app.models.database import db
from rich.console import Console
from rich.table import Table

console = Console()


async def test_apis():
    """Test all API connections"""
    
    console.print("\n[bold cyan]🧪 Testing API Connectivity[/bold cyan]\n")
    
    results = {}
    
    # Test Groq - USE SETTINGS MODEL
    console.print("[yellow]Testing Groq API...[/yellow]")
    try:
        client = GroqClient(settings.GROQ_API_KEY, settings.ATTACKER_MODEL)  # ✅ Fixed
        response = await client.generate_async("Say hello!", max_tokens=20)
        results["Groq"] = ("✅", f"OK - Response: {response[:30]}...")
        console.print(f"[green]✅ Groq API working (Model: {settings.ATTACKER_MODEL})[/green]\n")
    except Exception as e:
        results["Groq"] = ("❌", str(e)[:50])
        console.print(f"[red]❌ Groq API failed: {e}[/red]\n")
    
    # Test NVIDIA - USE SETTINGS MODEL
    console.print("[yellow]Testing NVIDIA API...[/yellow]")
    try:
        client = NVIDIAClient(settings.NVIDIA_API_KEY, settings.TARGET_MODEL_NAME)  # ✅ Fixed
        response = await client.generate_async("Say hello!", max_tokens=20)
        results["NVIDIA"] = ("✅", f"OK - Response: {response[:30]}...")
        console.print(f"[green]✅ NVIDIA API working (Model: {settings.TARGET_MODEL_NAME})[/green]\n")
    except Exception as e:
        results["NVIDIA"] = ("❌", str(e)[:50])
        console.print(f"[red]❌ NVIDIA API failed: {e}[/red]\n")
    
    # Test MongoDB
    console.print("[yellow]Testing MongoDB...[/yellow]")
    try:
        await db.connect()
        count = await db.get_collection_count("attacks")
        results["MongoDB"] = ("✅", f"OK - {count} attacks in DB")
        console.print(f"[green]✅ MongoDB working ({count} attacks)[/green]\n")
        await db.disconnect()
    except Exception as e:
        results["MongoDB"] = ("❌", str(e)[:50])
        console.print(f"[red]❌ MongoDB failed: {e}[/red]\n")
    
    # Summary table
    table = Table(title="API Test Results", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="white")
    
    for service, (status, details) in results.items():
        color = "green" if status == "✅" else "red"
        table.add_row(service, f"[{color}]{status}[/{color}]", details)
    
    console.print(table)
    
    # Overall
    all_passed = all(r[0] == "✅" for r in results.values())
    if all_passed:
        console.print("\n[bold green]🎉 All tests passed![/bold green]\n")
    else:
        console.print("\n[bold red]⚠️ Some tests failed. Check your configuration.[/bold red]\n")


if __name__ == "__main__":
    asyncio.run(test_apis())
