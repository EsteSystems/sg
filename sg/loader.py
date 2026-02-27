"""Gene loading — exec()-based dynamic loading.

Genes are Python source strings. Loading is exec() into a namespace
dict with gene_sdk injected. The gene's execute() function is
extracted and returned as a callable.
"""
