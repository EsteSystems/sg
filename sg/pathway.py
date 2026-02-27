"""Pathway execution — declared sequences of loci with late binding.

Pathways are fusion-aware: try the fused gene first, fall back to
decomposed steps. Supports composed pathways (->), iteration (for),
conditional binding (when), and dependency-based parallelism.
"""
