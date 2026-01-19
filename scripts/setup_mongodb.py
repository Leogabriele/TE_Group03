"""
MongoDB setup and initialization script
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.config import settings
from backend.app.models.database import db
from rich.console import Console

console = Console()


async def setup_database():
    """Setup MongoDB collections and indexes"""
    
    console.print("\n[bold cyan]🔧 MongoDB Setup[/bold cyan]\n")
    
    try:
        # Connect
        console.print("[yellow]Connecting to MongoDB...[/yellow]")
        await db.connect()
        console.print("[green]✅ Connected![/green]\n")
        
        # Collections
        collections = ["attacks", "responses", "evaluations", "metrics"]
        
        console.print("[yellow]Checking collections...[/yellow]")
        existing = await db.db.list_collection_names()
        
        for coll in collections:
            if coll in existing:
                count = await db.get_collection_count(coll)
                console.print(f"  ✅ {coll}: {count} documents")
            else:
                await db.db.create_collection(coll)
                console.print(f"  ✨ Created: {coll}")
        
        console.print("\n[green]✅ Database setup complete![/green]\n")
        
        # Display info
        console.print(f"[cyan]Database:[/cyan] {settings.MONGODB_DB_NAME}")
        console.print(f"[cyan]Collections:[/cyan] {len(collections)}")
        console.print()
        
    except Exception as e:
        console.print(f"\n[red]❌ Setup failed: {e}[/red]\n")
        raise
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(setup_database())
