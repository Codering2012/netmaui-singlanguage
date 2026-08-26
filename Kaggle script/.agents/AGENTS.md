### Empirical Code Audit & Anti-Hallucination Workflow
When presented with lists of bug claims, architectural critiques, or optimization proposals:
- **Audit Before Modifying**: Inspect authoritative source files and trace tensor shapes, parameters, and execution flow before making any edits.
- **Automated Self-Test Hypothesis Scripts**: Write dedicated, runnable hypothesis self-test scripts (e.g. in `scratch/test_claims.py`) to empirically test claims against live isolated tensors and functions before reaching a final verdict.
- **Mathematical & Code Debunking**: For claims that are invalid, explain precisely why they are false using mathematical proofs (e.g. broadcasting rules, tensor dimensions, derivative properties), direct code references, and empirical outputs from self-test scripts.
- **Surgical Fixes for Verified Bugs**: Implement precise fixes only for claims that are verified to be true bugs, preserving overall system contracts.
- **Mandatory Empirical Verification**: Always run compilation checks (`py_compile`/`pyflakes`) and execute automated unit tests to verify zero regressions before declaring success.
