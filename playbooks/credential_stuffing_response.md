# Playbook: Credential Stuffing Response

## Overview

Credential stuffing is an automated attack in which an adversary replays
username/password pairs obtained from unrelated data breaches against an
organization's authentication endpoints. The attack relies on password
reuse — a single successful login yields a validated, legitimate set of
credentials the attacker can escalate or monetize. Typical target surfaces
include SSO portals, VPN concentrators, webmail, and customer-facing
APIs. Distinguishing features from plain brute force are: (1) very low
per-account request rate, (2) very high unique-username diversity per
source IP, and (3) a high proportion of valid usernames mixed with
failures, reflecting that the wordlist comes from breached dumps.

## Detection Indicators

- Spike in authentication failures with HTTP 401/403 or SIEM auth-fail
  events across many distinct user accounts.
- Distributed source-IP set (often residential proxy networks) targeting
  the auth endpoint within a narrow time window.
- Low-entropy user-agent strings, headless browser fingerprints
  (`python-requests`, `Go-http-client`) or rotating but unusually similar
  UA patterns.
- Success on a small fraction of attempts followed by anomalous session
  behavior (new device, new geography, token-pair generation).
- Login attempts to default or privileged accounts (`admin`, `root`,
  `administrator`, service accounts).

## MITRE ATT&CK Mapping

- **T1110.004** — Brute Force: Credential Stuffing. Primary technique.
- **T1078** — Valid Accounts. Post-success persistence and lateral movement
  under a legitimate identity.
- **T1556** — Modify Authentication Process. Where MFA is bypassed via
  previously exfiltrated session cookies or consent phishing artifacts.

## Immediate Actions (first 15 minutes)

1. Apply a perimeter block on the top N source IPs at the WAF or edge
   firewall (ACL deny), prioritising IPs with the highest failure count
   and widest unique-username fan-out.
2. Enable or tighten rate limits on the auth endpoint — a hard cap of 5
   attempts per account per 5 minutes and per-IP caps below typical
   human rates (e.g., 30 auth requests/min).
3. Require step-up authentication (MFA challenge, email OTP) for all
   sessions that completed login from the attack window.
4. Disable or lock any accounts with a **successful** login from a
   flagged IP until the user re-verifies out of band.
5. Preserve raw auth logs and WAF captures from the window to a
   write-once location; the window is typically 2–6 hours longer than
   the visible spike.

## Investigation Steps

- Query the SIEM for successful logins from source IPs present in the
  flagged set, and for subsequent session activity (downloads, token
  creations, profile changes).
- Cross-reference failed usernames against internal user directories to
  identify which were valid; valid + failed is the best predictor of
  future success as the attacker rotates proxies.
- Correlate with external breach intelligence (HIBP, vendor feeds) —
  high overlap of valid usernames with a known breach list confirms
  credential stuffing over generic brute force.
- Check for concurrent session counts per user; legitimate users rarely
  hold more than two concurrent sessions, stuffing attackers do.
- Audit any API tokens or refresh tokens minted during the window and
  compare issuance source IP with user's historical pattern.

## Containment & Eradication

- For compromised accounts, force a password reset plus full session
  invalidation (revoke refresh tokens, clear persistent cookies,
  deauthorize OAuth grants).
- Roll any shared secrets the compromised user had access to:
  repository deploy keys, CI/CD secrets, API credentials.
- If an account successfully performed privileged actions, treat as a
  full intrusion and pivot to the broader IR plan — credential stuffing
  is an entry vector, not the payload.
- Update WAF signatures to include the observed UA patterns and the
  attacker's request timing signature for the next 14 days.

## Recovery

- Restore affected accounts with MFA enforced and a breached-password
  check against HIBP or equivalent at next password change.
- Notify users of the reset and the likely cause, following the
  organization's privacy policy and applicable regulation (GDPR,
  CCPA, 27001 A.16).
- Re-enable the auth endpoint with the tightened rate-limits and a
  CAPTCHA or risk-adaptive challenge for anonymous traffic.

## Post-Incident

- Deploy continuous breached-password screening for all authentication
  events and at password creation / rotation time.
- Enforce MFA for all privileged and externally-reachable accounts;
  prefer phishing-resistant factors (WebAuthn, FIDO2) over SMS or TOTP.
- Tune UEBA baselines to alert on impossible-travel and first-seen-user-agent
  events, not only on volume-based auth anomalies.
- Conduct a tabletop every 6 months covering credential-stuffing scenarios
  and verify the playbook's IPs-to-block and accounts-to-disable workflows
  can be executed within the 15-minute immediate-action window.
