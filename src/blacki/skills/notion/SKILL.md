---
name: notion
description: Interact with Notion workspaces - create, read, update, and search pages and databases.
version: 1.0.0
author: blacki
tags:
  - productivity
  - notes
  - database
---

# Notion MCP Skill

This skill provides access to Notion workspaces through the Notion MCP server.

## Prerequisites

Set the `NOTION_TOKEN` environment variable with your Notion integration token.

### Creating a Notion Integration

1. Go to https://www.notion.so/my-integrations
2. Click "New integration"
3. Give it a name and select your workspace
4. Copy the "Internal Integration Token"
5. In your Notion workspace, go to the pages/databases you want to share
6. Click "..." → "Add connections" → select your integration

## Pre-Discovered Resources

Use these IDs directly to avoid searching:

### Databases

| Database | ID | Parent Page |
|----------|----|-------------|
| Goals | `b414c187-e31c-431b-ba02-63edb5a19e78` | - |
| Tasks | `5b10261e-11cd-82ce-bc24-81b289fe381d` | Projects & Tasks |
| Projects | `c650261e-11cd-8208-b243-010047f73e84` | Projects & Tasks |
| To Read for investment | `11a52151-d36d-4a20-aaad-863b1fc8736d` | Reading List |

### Key Pages

| Page | ID |
|------|----|
| Projects & Tasks | `15c0261e-11cd-83f9-9fbc-81dbb75fa993` |
| Reading List | `373395b6-b07e-4291-a4fa-aa5d298c29dc` |
| Blog ideas | `b46b2d2a-0d12-4af4-90b5-990d5365a344` |
| Daily Entry +Life goals | `efb40f61-02fa-44b9-8f36-8335b8de5bc9` |
| Yearly Goals | `35f0f1b1-da67-4495-a104-7a901268a0fb` |

### Current Projects (In Progress)

| Project | ID | Description |
|---------|----|----|
| Garbanzo | `32c0261e-11cd-8151-9c5f-c15c1894e8df` | Personal AI assistant via Telegram |
| Spanning | `cbe0261e-11cd-8289-a27d-8195c82d5977` | Resume updater (90% complete) |
| Docs podcast | `31c0261e-11cd-80b7-851a-da3c624b1efe` | Documentation to podcast |

### Database Properties

**Tasks Database Properties:**
- Task name (title)
- Status: Not Started, In Progress, Done, Archived
- Priority: Low, Medium, High
- Tags: Mobile, Website, Improvement, Marketing, Research, Branding, Video production, Metrics
- Due (date)
- Project (relation to Projects)

**Projects Database Properties:**
- Project name (title)
- Status: Planning, In Progress, Paused, Backlog, Done, Canceled
- Priority: Low, Medium, High
- Dates (date range)
- Summary (text)
- Tasks (relation to Tasks)

**Goals Database Properties:**
- Name (title)
- Status (select)
- Due (select)
- Tags (multi_select)

**To Read Database Properties:**
- Name (title)
- Status: To read, Reading, Listening, Watching, Finished, Ready to Start
- Score /5: ⭐️ to ⭐️⭐️⭐️⭐️⭐️
- Type: Article, Film, Podcast, Academic Journal, TV Series, Book, etc.
- Author (multi_select)
- Publisher (select)
- Link (url)
- Summary (text)

## Usage Patterns

### Add a Task

```
notion_create_page with:
  parent: { database_id: "5b10261e-11cd-82ce-bc24-81b289fe381d" }
  properties: {
    "Task name": { title: [{ text: { content: "My new task" } }] },
    "Status": { select: { name: "Not Started" } },
    "Priority": { select: { name: "Medium" } }
  }
```

### Add a Project

```
notion_create_page with:
  parent: { database_id: "c650261e-11cd-8208-b243-010047f73e84" }
  properties: {
    "Project name": { title: [{ text: { content: "New Project" } }] },
    "Status": { select: { name: "Planning" } },
    "Priority": { select: { name: "High" } }
  }
```

### Query Tasks by Status

```
notion_query_database with:
  database_id: "5b10261e-11cd-82ce-bc24-81b289fe381d"
  filter: {
    property: "Status",
    select: { equals: "In Progress" }
  }
```

### Add to Reading List

```
notion_create_page with:
  parent: { database_id: "11a52151-d36d-4a20-aaad-863b1fc8736d" }
  properties: {
    "Name": { title: [{ text: { content: "Article Title" } }] },
    "Status": { select: { name: "To read" } },
    "Link": { url: "https://..." }
  }
```

## Tips

- Always use the pre-discovered IDs above to avoid searching
- Notion page and database IDs are UUIDs found in the URL
- Use markdown formatting in text content when appropriate
- Check property types before setting values (title, rich_text, number, select, etc.)
- For relations, use the page ID of the related item
