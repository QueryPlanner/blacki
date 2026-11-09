## ADDED Requirements

### Requirement: Private memory tools

Blacki SHALL expose wiki read, wiki write, and bounded history search to verified private Telegram owners by default, with an owner opt-out.

#### Scenario: A private owner uses memory without setup

- **WHEN** an enabled owner discusses a durable decision in private
- **THEN** Blacki can update the relevant wiki page with verified user evidence

#### Scenario: The caller has no verified owner

- **WHEN** a public, group, or synthetic turn attempts to use memory
- **THEN** it cannot read or write private memory

### Requirement: Sourced recall and review

Blacki SHALL distinguish original session evidence from derived wiki claims and use the three tools to ingest, query, and review the wiki.

#### Scenario: A user asks about a prior session

- **WHEN** the wiki lacks the requested detail
- **THEN** Blacki can search the owner's eligible retained sessions and cite an excerpt

#### Scenario: An owner authorizes a reminder

- **WHEN** a verified owner explicitly schedules a wiki review and still has access at delivery
- **THEN** Blacki can review the wiki with the three memory tools through the existing reminder service

### Requirement: Confirmed forgetting

Blacki SHALL require confirmation before deleting wiki claims and suppress their associated evidence from memory retrieval.

#### Scenario: Old evidence could recreate a claim

- **WHEN** an owner confirms forgetting a claim
- **THEN** the writer suppresses the known sources and requires a fresh explicit save to restore the fact
