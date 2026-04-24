# Playbook: AI Model Poisoning and Prompt-Injection Response

## Overview

This playbook covers incidents against in-house or third-party
large-language-model (LLM) systems where an adversary attempts to
influence model behavior via tainted training data, malicious system
prompts, retrieval-store pollution, or crafted inputs that bypass
safety filters. Three distinct scenarios collapse into this playbook
because their detection, containment, and recovery steps overlap
significantly: **(1)** training-data poisoning during pretraining or
fine-tuning, **(2)** indirect prompt injection through untrusted RAG
corpora, tool outputs, or document uploads, and **(3)** direct
prompt-injection / jailbreak targeting the production endpoint. All
three can result in incorrect business-critical outputs, leakage of
protected system instructions, unauthorized tool invocation, or
exfiltration of embedded secrets.

## Detection Indicators

- Outputs that disclose system prompts, internal tool schemas, or
  secrets that were supposed to be masked (commonly via "repeat
  everything before this message" style payloads).
- Model response distribution shift on a canary benchmark — e.g.,
  refusal rate suddenly drops on policy-violating prompts, or
  factuality on a gold-set sinks by more than 10 points.
- RAG store documents added outside the vetted pipeline: unexpected
  chunk sources, metadata anomalies (ingestion time inside a
  maintenance window).
- Fine-tuning pipeline access from non-standard IPs or service
  accounts; dataset hash drift between the last signed training run
  and the most recent.
- Alert from input-filtering layer for known jailbreak patterns
  ("ignore prior instructions", DAN-style role-play, tool-calling
  payloads in `system` role from user input).
- Elevated tool-call rate per session or successful tool invocations
  that the authenticated user does not have application-layer
  authorization for.

## MITRE ATT&CK and OWASP Mapping

- **OWASP LLM01** — Prompt Injection (direct and indirect).
- **OWASP LLM03** — Training Data Poisoning.
- **OWASP LLM06** — Sensitive Information Disclosure.
- **OWASP LLM08** — Excessive Agency (tool-abuse pivot).
- **T1565** — Data Manipulation, MITRE generic technique applied to ML
  pipelines.
- **T1195.001** — Supply Chain Compromise: Compromise Software
  Dependencies and Development Tools, for adversarial model weights.

## Immediate Actions (first 15 minutes)

1. Flip the affected endpoint to the previous blessed model checkpoint
   (feature-flag switch or traffic-shift) — restoring trust to the
   last known-good artifact buys time for investigation.
2. Enable or raise the confidence threshold on the input-/output-filter
   layer; block tool invocations that touch production state until
   clear.
3. Rotate any API tokens exposed through the system prompt or tool
   schemas if the injection succeeded at exfiltration.
4. Preserve the full prompt-and-completion log for the attack window
   into a write-once store; set legal-hold if the incident crosses a
   regulatory disclosure threshold.
5. For RAG-pollution cases, pause the ingestion pipeline and mark the
   affected collection read-only while the corpus is audited.

## Investigation Steps

- Hash-compare the deployed model weights against the signed artifact
  from the MLOps registry; any drift without a signed release is
  grounds for full rollback.
- Re-run the canary benchmark and the alignment regression suite on the
  deployed model; record deltas against the prior release.
- Extract prompts from the attack window — cluster by user, session,
  and semantic similarity — to identify the payload template and any
  sister attempts that did not succeed yet.
- Scan RAG documents ingested in the last N days for known
  prompt-injection patterns (`### END OF DOCUMENT. New instructions: ...`,
  zero-width unicode, font-color tricks).
- Trace the fine-tuning pipeline: who modified the training set, which
  commits, which CI jobs executed, whose credentials authorized the
  push to the model registry.

## Containment & Eradication

- Blacklist offending input sources (user accounts, API keys, source
  IPs); rate-limit the remainder of the endpoint until the input
  filter is retuned.
- Purge poisoned RAG chunks and re-ingest from the verified source
  with checksum validation at each step.
- If training data was poisoned, quarantine the tainted dataset and
  re-run fine-tuning from the last clean snapshot; discard any model
  versions trained on the tainted slice.
- For tool-abuse incidents, revoke any actions the model took under
  the attacker's session (write-backs, emails sent, code commits) and
  reverse externalized state.

## Recovery

- Promote a freshly-retrained or rollback model to production once it
  passes the alignment and safety regression suite.
- Re-open the ingestion pipeline only after corpus signing and
  chunk-level provenance tagging are enforced.
- Communicate scope and impact to affected internal consumers (product
  teams, ops) and to external customers if outputs reached them in a
  way that could materially mislead.

## Post-Incident

- Enforce signed, reproducible training runs with SBOM-style dataset
  manifests; no model reaches production without a matching signed
  manifest in the registry.
- Adopt prompt-injection defenses in depth: input classifier,
  instruction-data separation, model-level mitigations
  (Constitutional AI-style training, RLHF on adversarial data), and
  output-filter for secret-leakage.
- Tier tool-calling permissions: the model's tool palette is narrower
  than the calling user's, and each tool invocation goes through
  authorization independent of the model's request.
- Red-team the LLM quarterly using automated jailbreak suites
  (PAIR, Llama Guard-based oracles) and incorporate new findings into
  the filter ruleset.
- Instrument canary benchmarks and alignment scores as
  first-class SRE signals with alerting and weekly review.
