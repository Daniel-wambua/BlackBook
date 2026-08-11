"""BlackBook MCP — a source-grounded cybersecurity knowledge & research MCP server.

BlackBook is the *knowledge* teammate. It indexes a controlled corpus
(HackTricks, 0xdf writeups, local PDFs, ...) and answers questions like
"what is documented about this technique?" and "which similar cases exist?",
with exact, verifiable source provenance.

It is deliberately read-only with respect to external systems: it does not
execute commands, scan hosts, or exploit targets. Execution belongs to a
separate MCP (e.g. HexStrike); BlackBook informs the reasoning loop.
"""

__version__ = "0.6.0"

__all__ = ["__version__"]
