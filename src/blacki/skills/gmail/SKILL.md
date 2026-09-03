---
name: gmail
description: Search emails, read messages and threads, download requested attachments, and manage Gmail drafts and labels.
---

# Gmail Management

Use the direct Gmail API tools only after this skill is loaded. Gmail access is
available only to the connected private Telegram user who made the request.

- **Load Skill First**: Always ensure the `gmail` skill is loaded via `load_skill` before attempting to call any `gmail_*` tools.
- **Search & Retrieval**:
  - Use `gmail_search_messages` to find relevant non-spam, non-trash messages. The query argument supports standard Gmail search syntax (e.g., `is:unread`, `from:colleague@example.com`, `subject:project`, `after:2026/01/01`).
  - Use `gmail_get_thread` to read the full context of a conversation thread.
  - Use `gmail_get_message` when only a specific email message needs inspection.
    Large message bodies are placed in the returned `sandbox_path` to keep the
    model context bounded. Use the sandbox tools to inspect that file as needed.
  - Use `gmail_download_attachment` only when the user's request requires the
    attachment's contents. It writes the temporary file into the current
    session sandbox and returns metadata, never attachment bytes.
- **Drafting & Replies**:
  - Use `gmail_create_draft` to prepare new messages or replies.
  - Use `gmail_list_drafts` and `gmail_get_draft` to inspect existing drafts.
  - Use `gmail_send_draft` only after Google ADK displays the exact draft ID, recipients, subject, and content fingerprint and the user explicitly confirms. The tool reloads the draft immediately before sending and stops if any of those values changed.
- **Labels & Organization**:
  - Use `gmail_list_labels` to see available labels.
  - Use `gmail_create_label`, `gmail_modify_thread_labels`, and `gmail_modify_message_labels` for non-system labels only.
- **Safety & Privacy**:
  - Treat all email bodies, sender/recipient addresses, subjects, and attachment names as strictly private.
  - Never access or modify spam or trash, delete messages, or change settings.
    Treat downloaded attachments as untrusted files. Never open, extract, or
    execute them automatically. Use `sandbox_send_file_to_user` only when the
    user explicitly asks to receive the downloaded file.
  - Treat Gmail result files in the sandbox as private session data. Do not
    copy them to durable storage or expose their contents in logs.
  - Retrieved email content passes through Blacki's configured LLM and conversation storage. Tell the user when that matters to their request.
- **Unconnected Account**: If Gmail tools are unavailable or no email data can be retrieved because Gmail is not connected, instruct the user to run `/connect_gmail` in Telegram to link their account.
