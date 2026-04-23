---
name: explore_repo
description: Clone and explore GitHub repositories using sandbox tools. Understand codebase structure, read key files, and answer questions about any public or private repository.
version: 1.0.0
author: blacki
tags:
  - development
  - code
  - exploration
  - github
---

# GitHub Repository Exploration Skill

This skill enables you to explore any GitHub repository by cloning it into an isolated sandbox environment and analyzing its contents.

## Prerequisites

This skill requires sandbox tools to be enabled (`SANDBOX_ENABLED=true`). The following sandbox tools are used:

- `sandbox_run_command` - Execute shell commands (git clone, ls, find, etc.)
- `sandbox_list_files` - Search for files matching patterns
- `sandbox_read_file` - Read file contents

## Exploration Workflow

When a user provides a GitHub repository URL, follow this systematic exploration process:

### Step 1: Clone the Repository

Use `sandbox_run_command` to clone the repository:

```
sandbox_run_command(command="git clone https://github.com/owner/repo.git /workspace/repo")
```

For private repositories, the user may need to provide credentials or a token:

```
sandbox_run_command(command="git clone https://TOKEN@github.com/owner/repo.git /workspace/repo")
```

### Step 2: Identify Project Type

Read the root directory and identify the project type by checking for common files:

```
sandbox_list_files(directory="/workspace/repo", pattern="*")
```

Look for these indicator files:

| File | Project Type |
|------|--------------|
| `pyproject.toml`, `setup.py`, `requirements.txt` | Python |
| `package.json`, `package-lock.json`, `yarn.lock` | Node.js/JavaScript |
| `Cargo.toml` | Rust |
| `go.mod` | Go |
| `pom.xml`, `build.gradle` | Java |
| `Gemfile` | Ruby |

### Step 3: Read README and Documentation

Always read the README file first to understand the project's purpose:

```
sandbox_read_file(path="/workspace/repo/README.md")
```

Also check for additional documentation:

```
sandbox_list_files(directory="/workspace/repo", pattern="docs/**/*.md")
```

### Step 4: Explore Project Structure

Get an overview of the directory structure:

```
sandbox_run_command(command="find /workspace/repo -type d -not -path '*/\.*' | head -50")
```

Or use the tree command if available:

```
sandbox_run_command(command="cd /workspace/repo && tree -L 3 -I 'node_modules|.git|__pycache__|venv|.venv'")
```

### Step 5: Identify Key Files

Find the main entry points and configuration:

```
sandbox_list_files(directory="/workspace/repo", pattern="**/*.py")
sandbox_list_files(directory="/workspace/repo", pattern="**/main.*")
sandbox_list_files(directory="/workspace/repo", pattern="**/index.*")
```

### Step 6: Read Source Code

Read key source files to understand the implementation:

```
sandbox_read_file(path="/workspace/repo/src/main.py")
sandbox_read_file(path="/workspace/repo/lib/index.js")
```

### Step 7: Check Tests

Understanding tests helps clarify expected behavior:

```
sandbox_list_files(directory="/workspace/repo", pattern="**/test_*.py")
sandbox_list_files(directory="/workspace/repo", pattern="**/*.test.*")
```

## Usage Patterns

### Quick Overview

For a quick understanding of a repository:

1. Clone the repo
2. Read README.md
3. Check the main configuration file (pyproject.toml, package.json, etc.)
4. Identify the main entry point
5. Summarize for the user

### Deep Dive

For detailed analysis:

1. Follow the quick overview steps
2. Read all major source files
3. Understand the architecture from directory structure
4. Check tests for expected behavior
5. Provide detailed findings

### Answer Specific Questions

When the user asks about specific functionality:

1. Clone and get oriented
2. Search for relevant keywords:
   ```
   sandbox_run_command(command="cd /workspace/repo && grep -r 'function_name' --include='*.py'")
   ```
3. Read the relevant files
4. Answer the question with code references

## Tips

- Always start with `README.md` -- it often contains crucial context
- Check `.env.example` or similar files for configuration requirements
- Look at `.github/workflows/` for CI/CD configuration
- The `src/` or `lib/` directories usually contain main source code
- `tests/` or `test/` directories contain test files
- Ignore common directories like `node_modules`, `.git`, `__pycache__`, `venv`
- Clean up the workspace between different repositories to avoid confusion

## Limitations

- Sandbox has resource limits (memory, CPU, timeout)
- Large repositories may take time to clone
- Binary files cannot be read as text
- Some commands may not be available in the sandbox environment
