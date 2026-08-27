# Code Review Skill

## Overview

This skill performs automated code reviews for pull requests in the `pytorch-tcuscan` repository. It verifies that the code compiles successfully and passes all linters before providing a review.

## Pre-Review Checks

Before reviewing code changes, the following checks **must** pass:

### 1. Build Check

Ensure the project compiles cleanly:

```bash
make clean build
```

This removes any previous build artifacts and rebuilds the project from scratch. The review should fail if this command does not exit successfully.

### 2. Linter Check

Ensure all linters pass with no violations:

```bash
pre-commit run --all-files
```

This runs all configured pre-commit hooks (black, clang-format, cmake-format) across every file in the repository. The review should fail if any hook reports errors.

## Review Instructions

1. Run `make clean build` and report whether it succeeds or fails.
2. Run `pre-commit run --all-files` and report whether it succeeds or fails.
3. If either check fails, report the failure output and stop — do not proceed with the code review until both checks pass.
4. If both checks pass, proceed with a thorough review of the changed files, focusing on:
   - Correctness of logic and algorithms
   - Performance implications (especially for sparse matrix / NPU kernel code)
   - API consistency and backward compatibility
   - Test coverage for new functionality
   - Documentation accuracy
