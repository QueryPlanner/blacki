# Spec Delta

## Purpose

Defines how Blacki calls its OpenAI language model: the request and response
contract, how the model's reasoning carries across tool calls and turns, what
OpenAI may retain, and how failures surface.

## ADDED Requirements

### Requirement: OpenAI Responses transport
When the root model is an OpenAI model, every model call SHALL go directly to OpenAI's Responses API using the configured OpenAI API key, with no intermediary gateway.

#### Scenario: Model call goes to OpenAI
- **GIVEN** `OPENAI_API_KEY` is set and `ROOT_AGENT_MODEL` is a supported OpenAI model
- **WHEN** a Telegram turn needs a model response
- **THEN** the request is sent to OpenAI's Responses endpoint for that model
- **AND** no request is sent to OpenRouter

### Requirement: No server-side response storage
Every model request SHALL ask OpenAI not to store the response (`store: false`) and SHALL NOT reference a stored response by id.

#### Scenario: Request disables storage
- **WHEN** any model request is sent
- **THEN** it sets `store` to `false`
- **AND** it does not include `previous_response_id` or a conversation id

### Requirement: Conversation history is sent in full
Each request SHALL carry the turn history that the agent framework provides, translated into Responses input items: user text and images, the model's earlier output, tool calls, and tool results paired by call id.

#### Scenario: Tool result follows its call
- **GIVEN** the model called a tool in the previous step
- **WHEN** the next request is built
- **THEN** it contains that tool call and a matching tool result with the same call id, after it

#### Scenario: User image is sent
- **WHEN** a user turn contains an inline image
- **THEN** the request contains that image as image input for the user turn

#### Scenario: Calls without ids are paired by order
- **GIVEN** the framework supplies tool calls and results whose ids were cleared
- **WHEN** the request is built
- **THEN** each call and its result share one generated call id, paired in order

#### Scenario: Call without a result
- **GIVEN** a tool call has no matching result in the history
- **WHEN** the request is built
- **THEN** a placeholder tool result is added for it so the request is accepted

### Requirement: Reasoning replays unchanged to the same model
Output items the model returned, including reasoning items with their id, summary and encrypted content, SHALL be stored with the turn. They SHALL be sent back unchanged and in their original order whenever a later request goes to the same model.

#### Scenario: Reasoning carries across a tool call
- **GIVEN** a response returned a reasoning item followed by a tool call
- **WHEN** the request that carries the tool result is built
- **THEN** it contains that reasoning item, identical to what was returned, directly before the tool call

#### Scenario: Reasoning with no summary is kept
- **GIVEN** a response returned a reasoning item with encrypted content and an empty summary
- **WHEN** the next request to the same model is built
- **THEN** that reasoning item is still sent back

#### Scenario: Reasoning survives a restart
- **GIVEN** a session's turns were stored before the server restarted
- **WHEN** the session continues after the restart with the same model
- **THEN** the stored reasoning items are sent back as if no restart had happened

### Requirement: Reasoning is not sent to a different model
When a stored turn was produced by a different model than the one now being called, its reasoning items SHALL NOT be sent. Its visible text and tool calls SHALL still be sent in a portable form.

#### Scenario: Model changed between turns
- **GIVEN** earlier turns were produced by `gpt-6-luna`
- **WHEN** the chat switches to `gpt-6.1-terra` and a request is built
- **THEN** the request contains the earlier text and tool calls
- **AND** it contains no reasoning items from `gpt-6-luna`

### Requirement: Rejected reasoning does not fail the turn
If OpenAI rejects a request because of a replayed reasoning item, Blacki SHALL retry once without reasoning items. It SHALL stop replaying reasoning for that session for the rest of the process lifetime, and record a warning, instead of failing the user's turn.

#### Scenario: Stale reasoning is rejected
- **GIVEN** a replayed reasoning item is rejected by OpenAI as invalid
- **WHEN** the request fails with that client error
- **THEN** the same request is sent again without any reasoning items
- **AND** a warning that does not contain message content is logged

#### Scenario: Later requests skip reasoning
- **GIVEN** a session already had a reasoning rejection in this process
- **WHEN** the next request for that session is built
- **THEN** it contains no reasoning items and needs no failed attempt

#### Scenario: Unrelated client error
- **WHEN** a request fails with a 400 that does not point at a reasoning item
- **THEN** it is not retried and the error is raised

### Requirement: Orphaned reasoning is not sent
A replayed reasoning item that is not directly followed by an item from the same response SHALL be left out of the request.

#### Scenario: History around reasoning was changed
- **GIVEN** a stored reasoning item whose following item is no longer in the history
- **WHEN** the next request is built
- **THEN** that reasoning item is not sent

### Requirement: Reasoning effort and summaries use native controls
A chosen reasoning effort SHALL be sent as the Responses `reasoning.effort`, and a reasoning summary SHALL always be requested. Sampling parameters SHALL be sent only when effort is explicitly `none` and the model accepts them.

#### Scenario: Effort is applied
- **GIVEN** a chat selected effort `high`
- **WHEN** a request is built
- **THEN** it sets `reasoning.effort` to `high` and requests a reasoning summary

#### Scenario: No effort chosen
- **GIVEN** no effort is configured for the chat or the process
- **WHEN** a request is built
- **THEN** it omits `reasoning.effort` so the model's default applies
- **AND** it sends no `temperature` or `top_p`

### Requirement: Summaries are shown but never treated as answer text
Reasoning summaries SHALL be shown in Telegram as thinking. They SHALL NOT be sent or stored as user messages or as assistant answer text.

#### Scenario: Summary is displayed as thinking
- **WHEN** a response contains a reasoning summary and an answer
- **THEN** the summary is shown as thinking and the answer is shown as the reply

### Requirement: Commentary is not part of the reply
Assistant messages marked as commentary SHALL be kept in history and replayed, but SHALL NOT be included in the reply text sent to the user.

#### Scenario: Preamble before a tool call
- **GIVEN** a response contains a commentary message and a tool call, and a later response contains the final answer
- **WHEN** the turn's reply is assembled
- **THEN** the reply contains only the final answer

### Requirement: Refusals are shown, not retried
A model refusal SHALL be returned to the user as visible text and SHALL NOT be retried or routed elsewhere.

#### Scenario: Model refuses
- **WHEN** a response message contains refusal content
- **THEN** the refusal text is the reply and no retry is made

### Requirement: OpenAI data never reaches another provider
Stored OpenAI items and empty thought parts SHALL be removed from the request whenever the effective model is not an OpenAI model.

#### Scenario: Session continues on native Gemini
- **GIVEN** a session has turns produced by an OpenAI model
- **WHEN** a request for that session goes to native Gemini
- **THEN** the request contains no stored OpenAI items or encrypted reasoning

### Requirement: Compaction does not inject reasoning
When history is compacted, the stored summary SHALL contain only readable text. Its model call SHALL use the lowest reasoning effort above `none` and SHALL be recorded in the usage ledger.

#### Scenario: History is compacted
- **WHEN** compaction summarizes older events
- **THEN** the compacted turn contains no reasoning items or stored OpenAI items
- **AND** the summarizer call appears in the usage ledger

### Requirement: Encrypted reasoning stays out of logs and traces
Encrypted reasoning content and raw stored output items SHALL NOT be written to application logs, at any level, to exported trace attributes, or to the usage ledger. Readable reasoning summaries MAY appear wherever message content capture is already enabled.

#### Scenario: Trace export with content capture on
- **GIVEN** trace message-content capture is enabled
- **WHEN** a turn with replayed reasoning is traced
- **THEN** exported span attributes contain no `encrypted_content` values

#### Scenario: Debug logging
- **GIVEN** debug logging is enabled
- **WHEN** a request with replayed reasoning is logged
- **THEN** the log output contains no `encrypted_content` values

#### Scenario: Legacy span capture requested
- **GIVEN** an operator enables the framework's legacy full-content span capture
- **WHEN** the server starts
- **THEN** that capture stays disabled and a warning explains why

### Requirement: Streaming yields partial output and one final turn
For streamed turns, Blacki SHALL yield partial answer text and partial reasoning summaries as they arrive. It SHALL then yield exactly one final response holding the complete output items, usage and finish reason.

#### Scenario: Streamed turn completes
- **WHEN** a streamed response delivers text, a reasoning summary and a tool call
- **THEN** partial text and summary updates are yielded during the stream
- **AND** one final response with all output items and usage is yielded at the end

### Requirement: Provider failures surface clearly
A failed model call SHALL raise a typed error carrying the HTTP status and OpenAI's error code. It SHALL NOT contain the API key or prompt content.

#### Scenario: Authentication failure
- **WHEN** OpenAI returns HTTP 401
- **THEN** the turn fails with a provider authentication error that names the status and contains no key or prompt text

#### Scenario: Output limit reached
- **WHEN** a response ends incomplete because of the output token limit, streamed or not
- **THEN** exactly one final response is produced and reports a max-tokens finish reason

#### Scenario: Content filter
- **WHEN** a response ends incomplete because of the content filter
- **THEN** the final response reports a safety finish reason

### Requirement: Missing configuration fails at startup
If an OpenAI model is configured without `OPENAI_API_KEY`, startup SHALL fail with an error that names the missing setting. An unsupported model id SHALL be rejected with the list of supported ids.

#### Scenario: Key missing
- **GIVEN** `ROOT_AGENT_MODEL` is an OpenAI model and `OPENAI_API_KEY` is unset
- **WHEN** the server starts
- **THEN** startup fails with an error naming `OPENAI_API_KEY`

#### Scenario: Native Gemini still works
- **GIVEN** `ROOT_AGENT_MODEL` is a native Gemini model and `GOOGLE_API_KEY` is set
- **WHEN** the server starts
- **THEN** the agent uses native Gemini and requires no OpenAI key
