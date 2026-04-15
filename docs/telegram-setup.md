# Telegram Bot Setup Guide

This guide walks you through setting up a Telegram bot for the blacki agent.

## Prerequisites

- A Telegram account
- Access to the blacki server environment

## Step 1: Create a Telegram Bot

1. **Open Telegram** and search for **@BotFather** (the official Telegram bot for creating bots)

2. **Start a conversation** with @BotFather by clicking "Start" or sending `/start`

3. **Create a new bot** by sending the command:
   ```
   /newbot
   ```

4. **Choose a name** for your bot (this is the display name, e.g., "Blacki AI Assistant")

5. **Choose a username** for your bot (must end with `bot`, e.g., `blacki_ai_bot`)

6. **Copy the API token** that @BotFather gives you. It looks like:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

   ⚠️ **Keep this token secret!** Anyone with this token can control your bot.

## Step 2: Configure the Bot

Add the token to your `.env` file:

```bash
# Telegram Configuration
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

## Step 3: Deploy

After updating the `.env` file:

```bash
# If running locally
uv run python -m blacki.server

# If running with Docker
docker compose up -d
```

## Step 4: Test the Bot

1. **Find your bot** on Telegram by searching for the username you chose
2. **Start a conversation** by clicking "Start" or sending `/start`
3. **Send a message** and verify the bot responds

## Bot Commands

The blacki Telegram bot supports these commands:

| Command | Description |
|---------|-------------|
| `/start` | Start a conversation and see welcome message |
| `/help` | Show available commands and usage |
| `/clear` | Clear conversation memory for fresh start |

## Memory Features

The bot uses **mem0** for persistent memory:

- Each Telegram chat has its own isolated memory context
- The bot remembers previous conversations with each user
- Memories are stored locally in `./data/qdrant`

### Testing Memory

1. Tell the bot something about yourself: "My name is Alice and I like Python"
2. Later, ask: "What's my name?" or "What programming language do I like?"
3. The bot should remember from the previous conversation

## Security Considerations

### Token Security

- **Never commit** the bot token to git
- Use environment variables or secret management
- If the token is compromised, regenerate it via @BotFather:
  ```
  /mybots
  → Select your bot
  → API Token
  → Revoke Token
  ```

### Bot Privacy

By default, bots can only see:
- Messages that start with `/` (commands)
- Messages where the bot is mentioned
- Messages in groups where the bot has privacy mode disabled

For 1-on-1 conversations, the bot sees all messages.

## Troubleshooting

### Bot Not Responding

1. **Check the logs:**
   ```bash
   docker compose logs -f
   ```

2. **Verify the token:**
   - Make sure `TELEGRAM_ENABLED=true`
   - Check that the token is correct in `.env`

3. **Check network connectivity:**
   - The bot needs to reach Telegram's API servers
   - If behind a firewall, ensure outbound HTTPS is allowed

### Bot Responds with Errors

1. **Check LLM configuration:**
   - Verify `OPENROUTER_API_KEY` is set
   - Verify `ROOT_AGENT_MODEL` is correct

2. **Check database:**
   - Verify `DATABASE_URL` is correct
   - Ensure the database is accessible

### Memory Not Working

1. **Check mem0 configuration:**
   - `OPENROUTER_API_KEY` must be set for mem0's LLM operations
   - Check that `./data/qdrant` directory is writable

2. **View memory storage:**
   ```bash
   ls -la ./data/qdrant
   ```

## Advanced Configuration

### Custom Model for Telegram

To use a different model for Telegram responses, you can modify the bot's `model` parameter in `server.py`.

### Webhook Mode (Production)

For high-traffic bots, consider using webhooks instead of polling:

1. Set up a domain with HTTPS
2. Configure `TELEGRAM_WEBHOOK_URL` in `.env`
3. Modify `bot.py` to use webhook mode

Webhook mode is more efficient but requires:
- A public domain
- Valid SSL certificate
- Port 443 accessible from the internet

## Rate Limits

Telegram has rate limits for bots:
- ~30 messages/second to the same group
- ~1 message/second to the same user

The bot handles these automatically with the `python-telegram-bot` library.
