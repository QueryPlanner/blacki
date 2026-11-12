# Spec Delta

## Purpose

Defines which OpenAI models Blacki supports, and how the process default and each
Telegram chat choose a model and reasoning effort, including how saved legacy
selections are read.

## ADDED Requirements

### Requirement: Supported model catalog
Blacki SHALL keep a catalog of supported OpenAI models. Each entry lists the model id, display name, accepted reasoning efforts, default effort and token prices. Only catalog models can be selected.

#### Scenario: Catalog lists GPT-6 models
- **WHEN** the catalog is read
- **THEN** it contains `gpt-6-luna` and `gpt-6.1-terra`, each with its accepted efforts and default effort

### Requirement: Process default model
`ROOT_AGENT_MODEL` SHALL accept a plain catalog model id. For compatibility, a value of the form `openrouter/openai/<id>` SHALL be read as `<id>`.

#### Scenario: Plain id
- **GIVEN** `ROOT_AGENT_MODEL=gpt-6.1-terra`
- **WHEN** the server starts
- **THEN** the root agent uses `gpt-6.1-terra`

#### Scenario: Legacy OpenRouter OpenAI id
- **GIVEN** `ROOT_AGENT_MODEL=openrouter/openai/gpt-6-luna`
- **WHEN** the server starts
- **THEN** the root agent uses `gpt-6-luna` and logs that the legacy form was converted

#### Scenario: Unsupported model
- **GIVEN** `ROOT_AGENT_MODEL=openrouter/anthropic/claude-x`
- **WHEN** the server starts
- **THEN** startup fails with an error listing the supported model ids

### Requirement: Chat model selection
`/model <id>` SHALL save a catalog model for later turns in that chat and reset the chat's effort to the model default. `/model default` SHALL clear the selection. An unknown id SHALL be rejected with the supported ids.

#### Scenario: Valid selection
- **WHEN** a user sends `/model gpt-6-luna`
- **THEN** later turns in that chat use `gpt-6-luna`
- **AND** the reply confirms the model and says thinking uses the default setting

#### Scenario: Unknown model
- **WHEN** a user sends `/model some/other-model`
- **THEN** nothing is saved and the reply lists the supported model ids

#### Scenario: OpenAI not configured
- **GIVEN** the root model is native Gemini and `OPENAI_API_KEY` is unset
- **WHEN** a user sends `/model gpt-6-luna`
- **THEN** nothing is saved and the reply says OpenAI is not configured

### Requirement: Thinking options follow the selected model
The `/thinking` menu SHALL offer only the efforts the effective model accepts, plus "default". Choosing an effort the model does not accept SHALL be refused.

#### Scenario: Effort menu for a model without none
- **GIVEN** the effective model does not accept effort `none`
- **WHEN** a user opens `/thinking`
- **THEN** `none` is not offered

#### Scenario: Stale effort is refused
- **GIVEN** a thinking button for an effort the current model does not accept
- **WHEN** the user presses it
- **THEN** the setting is not saved and the user is told the option is unavailable

### Requirement: Saved legacy selections are read safely
A saved selection of the form `openrouter/openai/<id>` SHALL be used as `<id>` when `<id>` is in the catalog and OpenAI is configured. Any other saved selection that is not a usable catalog id SHALL be ignored in favor of the default model. Neither case SHALL fail the turn.

#### Scenario: Legacy OpenAI selection
- **GIVEN** a chat saved `openrouter/openai/gpt-6-luna`
- **WHEN** the chat's next turn runs
- **THEN** it uses `gpt-6-luna`

#### Scenario: Legacy non-OpenAI selection
- **GIVEN** a chat saved `openrouter/google/gemini-2.5-flash`
- **WHEN** the chat's next turn runs
- **THEN** it uses the process default model and logs a warning

#### Scenario: Saved plain Gemini id with an OpenAI root
- **GIVEN** the root model is an OpenAI model and a chat saved `gemini-2.5-flash`
- **WHEN** the chat's next turn runs
- **THEN** it uses the process default model and logs a warning

### Requirement: Saved effort is validated against the model
When a saved effort is not accepted by the effective model, the model's default effort SHALL be used instead of sending an unsupported value.

#### Scenario: Saved effort unsupported after a model change
- **GIVEN** a chat saved effort `none` and the effective model does not accept `none`
- **WHEN** the chat's next turn runs
- **THEN** the request uses the model's default effort and a warning is logged
