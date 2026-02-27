"""Safety mechanisms — transactions, blast radius, rollback.

Wraps configuration gene execution in undo-log transactions.
Classifies loci by blast radius (none → low → medium → high → critical)
and applies graded safety policies: shadow mode, canary, convergence
checks, resilience requirements.
"""
