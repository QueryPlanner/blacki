# Spec Delta

## Purpose

Defines how Blacki records each model call's token usage and estimated cost
when the provider reports tokens but no cost, so spending stays visible on the
dashboard and in traces.

## ADDED Requirements

### Requirement: Token usage is recorded per model call
Each completed model call SHALL record input, cached input, cache-write and output tokens in the usage ledger. The model span SHALL also carry the reasoning token count.

#### Scenario: Usage recorded
- **WHEN** a model call completes with reported usage
- **THEN** the ledger row carries its input, cached input, cache-write and output token counts
- **AND** the model span carries its reasoning token count

### Requirement: Estimated cost from catalog prices
When the provider reports no cost, Blacki SHALL record an estimated cost from reported tokens and catalog prices. Cached input uses the cached rate and other input the input rate. Reasoning tokens are part of output tokens and are not added again. The record SHALL be marked as an estimate.

#### Scenario: Estimate computed
- **GIVEN** a call reports 10,000 input tokens of which 4,000 were cached, and 1,000 output tokens
- **WHEN** the call is recorded
- **THEN** the estimate equals 6,000 input-rate tokens plus 4,000 cached-rate tokens plus 1,000 output-rate tokens
- **AND** the record's cost kind is estimate

#### Scenario: Reasoning tokens not double-counted
- **GIVEN** a call reports 1,000 output tokens of which 600 are reasoning tokens
- **WHEN** the call is recorded
- **THEN** exactly 1,000 tokens are priced at the output rate

#### Scenario: Cache writes are priced when reported
- **GIVEN** a call's usage includes a cache-write token count and the model has a cache-write price
- **WHEN** the call is recorded
- **THEN** those tokens are priced at the cache-write rate instead of the input rate

#### Scenario: Cache writes not reported
- **GIVEN** a call's usage has no cache-write token count
- **WHEN** the call is recorded
- **THEN** all uncached input is priced at the input rate

### Requirement: Unknown prices never invent a cost
If a model has no catalog price, Blacki SHALL record tokens with no cost and log a warning once per model, instead of estimating.

#### Scenario: Model without prices
- **GIVEN** a model call for a model with no catalog price
- **WHEN** the call is recorded
- **THEN** token counts are stored and the cost fields are empty

### Requirement: Streaming records cost once
A streamed model call SHALL produce exactly one usage record, taken from the final response.

#### Scenario: Streamed call
- **WHEN** a streamed call yields several partial responses and one final response
- **THEN** exactly one ledger record is written for that call
