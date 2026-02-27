"""Fitness scoring, promotion, and demotion.

Temporal fitness: immediate (t=0, 30%) + convergence (t=30s, 50%)
+ resilience (t=hours, 20%). Retroactive decay when convergence
or resilience checks fail.
"""
