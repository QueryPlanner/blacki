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

## Pre-Discovered Resources

### Account Info

- **Username:** QueryPlanner
- **Name:** VectorQL

### Active Repositories (by recent activity)

| Repository | Description | Visibility |
|------------|-------------|------------|
| `QueryPlanner/blacki` | - | public |
| `QueryPlanner/upgraded-garbanzo` | Personal AI assistant (Telegram) | private |
| `QueryPlanner/biodata` | biodata | private |
| `QueryPlanner/latex-resume-ai` | Create resumes from Job description | private |
| `QueryPlanner/Binge-Docs` | Turn docs into personal podcast | public |
| `QueryPlanner/google-adk-on-bare-metal` | Deploy google adk on bare metal | public |
| `QueryPlanner/qwentts-fastapi` | FastAPI server for Qwen3-TTS | public |
| `QueryPlanner/Chatter-Fast-Chatter-Box` | FastAPI for Chatterbox TTS with voice cloning | public |
| `QueryPlanner/adk-samples` | Sample agents built with ADK | public |
| `QueryPlanner/blogs` | - | private |
| `QueryPlanner/open-recite` | Recite anything - learn fast | public |
| `QueryPlanner/browser-text-api` | FastAPI wrapper for article text extraction | private |
| `QueryPlanner/ThreadX` | YouTube to X thread automation | private |
| `QueryPlanner/valuation-pro` | - | public |
| `QueryPlanner/blog-agent` | - | public |
| `QueryPlanner/medical-agents` | Production ready medical agents | public |
| `QueryPlanner/adk-extra-services` | Additional services for Google ADK | public |
| `QueryPlanner/GenAI` | All GenAI explorations | public |
| `QueryPlanner/research-paper-finder` | Find top research papers on any topic | public |

### Project Repositories

| Project | Repository | Notes |
|---------|------------|-------|
| Current Project | `QueryPlanner/blacki` | ADK on bare metal template |
| Garbanzo | `QueryPlanner/upgraded-garbanzo` | Personal AI assistant |
| Resume AI | `QueryPlanner/latex-resume-ai` | Resume from JD |
| Binge Docs | `QueryPlanner/Binge-Docs` | Docs to podcast |
| ThreadX | `QueryPlanner/ThreadX` | YouTube to X thread |

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
- Use the pre-discovered repository names above to avoid listing/searching
