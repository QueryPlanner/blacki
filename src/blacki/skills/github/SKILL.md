---
name: github
description: Interact with GitHub - manage repositories, issues, pull requests, and more.
version: 1.0.0
author: blacki
tags:
  - development
  - code
  - collaboration
---

# GitHub MCP Skill

This skill provides access to GitHub through the GitHub MCP server.

## Prerequisites

Set the `GITHUB_TOKEN` environment variable with a GitHub Personal Access Token.

### Creating a GitHub Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a descriptive name
4. Select the scopes you need:
   - `repo` - Full control of private repositories
   - `read:org` - Read org and team membership
   - `read:user` - Read user profile data
5. Click "Generate token" and copy it immediately

## Discovering Resources

Use `github_list_repositories` to find repositories you have access to. Repository names follow the `owner/repo` format.

### Common Operations

- `github_list_issues` - List issues in a repository
- `github_create_issue` - Create a new issue
- `github_create_pull_request` - Create a pull request
- `github_get_file_contents` - Read file contents
- `github_create_or_update_file` - Create or modify files

## Usage Patterns

### Working with Issues

1. Use `github_list_issues` to find issues in a repository
2. Use `github_create_issue` to create a new issue
3. Use `github_add_issue_comment` to add comments

### Creating Pull Requests

1. Ensure your branch exists with the desired changes
2. Use `github_create_pull_request` with source and target branches
3. Optionally request reviewers and add labels

### Working with Files

1. Use `github_get_file_contents` to read file contents
2. Use `github_create_or_update_file` to create or modify files
3. Always provide a commit message for file changes

## Tips

- Repository names are in `owner/repo` format
- Use branch names like `main` or `master` for the default branch
- Reference issues with `#123` syntax in commit messages and PR descriptions
- Use markdown formatting in issue and PR descriptions
