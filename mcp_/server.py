import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

NOTES_FILE = os.path.join(os.path.dirname(__file__), 'notes.txt')

load_dotenv()

###############################################################################
# Server
###############################################################################
mcp = FastMCP('Everysk MCP Server')

def ensure_file():
    """Ensure the notes file exists."""
    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, 'w') as f:
            f.write('')

@mcp.tool()
def add_note(note: str) -> str:
    """
    Add a note to the notes file.

    Args:
        note (str): The note to add.

    Returns:
        str: Confirmation message.
    """
    ensure_file()
    with open(NOTES_FILE, 'a') as f:
        f.write(note + '\n')
    return f'Note added: {note}'

@mcp.tool()
def read_notes() -> str:
    """
    Read all notes from the notes file.

    Returns:
        str: All notes.
    """
    ensure_file()
    with open(NOTES_FILE, 'r') as f:
        notes = f.read().strip()
    return notes or 'No notes found.'

@mcp.tool()
def get_portfolio(workspace: str, tags: str) -> list[dict]:
    """
    Get the portfolio from Everysk based on workspace and tags.

    Args:
        workspace (str): The workspace to query.
        tags (str): The tags to filter by.

    Returns:
        list[dict]: The portfolio data.
    """
    from everysk.sdk.entities import Portfolio
    portfolio = Portfolio.query.where('workspace', workspace).where('tags', tags).load()
    return portfolio

@mcp.resource('notes://latest')
def get_latest_note() -> str:
    """
    Get the latest note from the notes file.

    Returns:
        str: The latest note.
    """
    ensure_file()
    with open(NOTES_FILE, 'r') as f:
        notes = f.readlines()
    return notes[-1].strip() if notes else 'No notes found.'

@mcp.prompt()
def note_summary_prompt() -> str:
    """
    Prompt for a summary of the notes.

    Returns:
        str: Summary of the notes.
    """
    ensure_file()
    with open(NOTES_FILE, 'r') as f:
        notes = f.read().strip()
    if not notes:
        return 'No notes found.'
    
    return f'Summarize the current notes: {notes}'
