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

## Discovering Resources

Use `notion_search` to find databases and pages in your workspace. Notion IDs are UUIDs found in the URL when viewing a page or database.

### Common Database Operations

- `notion_query_database` - Query a database with optional filters
- `notion_create_page` - Create a new page in a database
- `notion_update_page` - Update an existing page

### Finding Database IDs

1. Navigate to your database in Notion
2. The ID is in the URL: `notion.so/YOUR_WORKSPACE/DATABASE_ID?v=...`
3. Copy the UUID portion (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

## Usage Patterns

### Add a Task

```
notion_create_page with:
  parent: { database_id: "YOUR_DATABASE_ID" }
  properties: {
    "Task name": { title: [{ text: { content: "My new task" } }] },
    "Status": { select: { name: "Not Started" } },
    "Priority": { select: { name: "Medium" } }
  }
```

### Query Tasks by Status

```
notion_query_database with:
  database_id: "YOUR_DATABASE_ID"
  filter: {
    property: "Status",
    select: { equals: "In Progress" }
  }
```

### Add to Reading List

```
notion_create_page with:
  parent: { database_id: "YOUR_DATABASE_ID" }
  properties: {
    "Name": { title: [{ text: { content: "Article Title" } }] },
    "Status": { select: { name: "To read" } },
    "Link": { url: "https://..." }
  }
```

## Tips

- Use `notion_search` to discover available databases and pages
- Notion page and database IDs are UUIDs found in the URL
- Use markdown formatting in text content when appropriate
- Check property types before setting values (title, rich_text, number, select, etc.)
- For relations, use the page ID of the related item
