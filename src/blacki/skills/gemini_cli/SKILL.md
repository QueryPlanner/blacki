---
name: gemini_cli
description: Reference for a disabled Gemini CLI sandbox capability.
---

# Gemini CLI Skill

The Gemini CLI (`@google/gemini-cli`) is a command-line interface for interacting with the Google Gemini API.
Blacki does not register this skill. General-purpose sandboxes intentionally
receive no model or repository credentials, so enabling the commands below would
create a non-functional or unsafe capability.

## Pre-requisites
Do not enable this skill by injecting a standing API key. A future implementation
must use an explicit, least-privilege capability broker with short-lived
credentials and separate allow/deny tests.

For reference, the Gemini CLI installation command is:
```bash
hash gemini 2>/dev/null || npm install -g @google/gemini-cli@latest
```

## Usage
Since you are an automated agent, you MUST use the Gemini CLI in non-interactive mode.

### Non-Interactive Query
Use the `-p` (or `--prompt`) flag to send a query non-interactively. This will execute the task and return the output without dropping into the REPL.

```bash
sandbox_run_command(command='gemini -p "Summarize the README.md file in the current directory."')
```

### Passing Context
You can pipe file contents into Gemini for context, or rely on its ability to read the workspace.

```bash
sandbox_run_command(command='cat main.py | gemini -p "Explain this code and suggest refactoring."')
```

### Resume Session
You can resume a previous session if you need to continue a conversation.

```bash
sandbox_run_command(command='gemini -r "latest" "Apply the suggested refactoring to the code."')
```

# Gemini CLI cheatsheet

This page provides a reference for commonly used Gemini CLI commands, options,
and parameters.

## CLI commands

| Command                            | Description                        | Example                                                      |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| `gemini -p "query"`                | Query non-interactively            | `gemini -p "summarize README.md"`                            |
| `cat file \| gemini`               | Process piped content              | `cat logs.txt \| gemini`<br>`Get-Content logs.txt \| gemini` |
| `gemini -r "latest"`               | Continue most recent session       | `gemini -r "latest"`                                         |
| `gemini -r "latest" "query"`       | Continue session with a new prompt | `gemini -r "latest" "Check for type errors"`                 |
| `gemini update`                    | Update to latest version           | `gemini update`                                              |

### Positional arguments

| Argument | Type              | Description                                                                                                |
| -------- | ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `query`  | string (variadic) | Positional prompt. Defaults to interactive mode in a TTY. Use `-p/--prompt` for non-interactive execution. |

## CLI Options

| Option                           | Alias | Type    | Default   | Description                                                                                                                                                            |
| -------------------------------- | ----- | ------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`                        | `-m`  | string  | `auto`    | Model to use. See [Model Selection](#model-selection) for available values.                                                                                            |
| `--prompt`                       | `-p`  | string  | -         | Prompt text. Appended to stdin input if provided. Forces non-interactive mode.                                                                                         |
| `--resume`                       | `-r`  | string  | -         | Resume a previous session. Use `"latest"` for most recent or index number (for example `--resume 5`)                                                                   |
| `--output-format`                | `-o`  | string  | `text`    | The format of the CLI output. Choices: `text`, `json`, `stream-json`                                                                                                   |
