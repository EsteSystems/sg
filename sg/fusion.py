"""Pathway fusion — reinforcement tracking, fuse/decompose cycle.

When a pathway executes successfully 10 consecutive times with the
same allele composition, it fuses into a single optimized gene.
If the fused gene fails, it decomposes back to individual steps.
"""
