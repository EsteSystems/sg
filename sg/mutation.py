"""Mutation engines — mock (fixtures) and Claude (API).

When all alleles are exhausted, the orchestrator triggers mutation:
an LLM generates a new allele from the contract, failing source,
error context, diagnostic output, and environmental state.
"""
