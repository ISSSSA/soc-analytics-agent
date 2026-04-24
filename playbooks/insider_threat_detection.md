# Playbook: Insider Threat Detection and Response

## Overview

Insider threats cover malicious, negligent, or compromised behavior by
people holding legitimate credentials — employees, contractors, or
third-party vendors with sustained access to internal systems. Because
the activity occurs under valid identities, perimeter controls are
usually not the point of detection; instead, user-and-entity behavior
analytics (UEBA) and privileged-access reviews surface the anomaly.
This playbook applies whether the driver is personal (grievance,
financial pressure, pending departure), ideological (whistleblowing,
activism), or external (an attacker has subverted the employee's
account via phishing or a malicious browser extension).

## Detection Indicators

- Unusually large data egress from an authenticated user — measured by
  bytes to personal cloud storage, email-to-personal-address patterns,
  or USB write volumes on managed endpoints.
- Access to repositories, S3 buckets, or database tables outside the
  user's role-based responsibilities, especially during off-hours or
  from previously-unseen devices or geographies.
- Privilege enumeration commands — `whoami`, `id`, `az role assignment
  list`, `aws iam list-attached-role-policies`, `Get-ADGroupMember` —
  issued by a user who has no administrative function.
- Mass download events from source control (git clone of many
  repos in quick succession), design systems, or CRM exports above
  historical baseline.
- Credential-sharing patterns: concurrent sessions from
  geographically-distant IPs, shared SSH keys used by multiple users,
  service accounts authenticated from interactive workstations.
- HR signals correlated in time: the user is on a performance
  improvement plan, has submitted a resignation, or has had an
  unfavorable review in the last 30 days.

## MITRE ATT&CK Mapping

- **T1078** — Valid Accounts. The core technique — authenticated but
  unauthorized-by-intent activity.
- **T1087** — Account Discovery. Enumeration of users, groups, and
  permissions.
- **T1530** — Data from Cloud Storage Object. Frequent exfiltration
  vector in modern environments.
- **T1213** — Data from Information Repositories. Includes git,
  Confluence, SharePoint, ticketing systems.
- **T1114** — Email Collection, when the user forwards or auto-rules
  mail to external addresses.
- **T1537** — Transfer Data to Cloud Account, when exfiltration uses
  legitimate cloud APIs to attacker-controlled storage.

## Immediate Actions (first 15 minutes)

1. Engage HR and Legal before taking containment actions that affect
   the individual — insider-threat handling has employment-law and
   evidence-preservation requirements beyond pure IR.
2. Snapshot the user's sessions, endpoint state, and the last 30 days
   of authentication and file-access logs into a tamper-evident store.
   Do not modify the user's state until the snapshot completes.
3. With HR and Legal approval, invalidate all active sessions for the
   user and revoke MFA devices, API tokens, OAuth grants, and
   SSO sessions.
4. Disable (do not delete) the user's identity in the directory,
   preserving the object for forensic correlation.
5. Revoke physical access (badge, VPN cert) and notify the user's
   manager on a need-to-know basis, per the IR communication plan.

## Investigation Steps

- Produce a time-ordered activity timeline from SIEM for the target
  user covering the last 90 days: authentication, file access, data
  egress, privileged-action events, and HR events.
- Review the UEBA risk score trajectory — sustained rises
  (not one-off spikes) over days or weeks are strong signals; pair
  with the volume and sensitivity of the data accessed.
- Enumerate the data sets the user had access to and compute which of
  those were touched, copied, printed, or emailed in the incident
  window. Rank by classification level (PII, IP, financial, source).
- Interview peers and managers under Legal/HR guidance to establish
  whether the behavior aligns with a legitimate business need.
- Where a compromised-account hypothesis is active, examine the user's
  endpoint for unauthorized software, browser extensions, or
  malware; consider imaging and submitting to Threat Intel.

## Containment & Eradication

- Rotate any shared secrets the user had access to: deploy keys,
  service credentials, shared mailbox passwords, Wi-Fi pre-shared keys,
  and privileged-account credentials stored in the password vault.
- Audit and revoke external shares (Google Drive, OneDrive,
  Dropbox) created or modified by the user in the window.
- Disable automations the user owned (scheduled tasks, cron jobs,
  CI/CD pipelines, Lambda/Cloud Functions) until each is reviewed and
  re-owned by another accountable party.
- Ensure outbound data-loss-prevention (DLP) rules now block the
  patterns observed (specific email recipients, specific cloud
  destinations, specific archive formats with encrypted content).

## Recovery

- Complete off-boarding, remediation, or re-authentication per HR
  decision; return the user's identity to the appropriate final
  state (disabled, deleted after retention period, or reinstated
  with an adjusted access scope).
- Restore any altered data from backup where tampering is confirmed,
  and revoke external distributions where feasible (DMCA, legal
  takedown, platform-level revocation).
- Document the complete timeline for regulators and the incident
  register; preserve for the standard legal-hold retention period.

## Post-Incident

- Deploy or tune UEBA with rules for the behavioral signatures seen:
  impossible travel, first-seen-device, data-egress spikes, after-hours
  privileged access.
- Reduce standing privilege with just-in-time (JIT) access where
  technically feasible; default-deny plus short-lived elevation beats
  persistent admin membership.
- Build a joint HR–Security insider-risk program that integrates HR
  events (resignations, PIPs, disputes) into the SIEM as low-fidelity
  risk signals that tune UEBA sensitivity for the affected user.
- Review role-based access grants org-wide for excessive scope; focus
  on roles that let users touch data far beyond their job function.
- Conduct an annual insider-threat tabletop with HR, Legal, and
  Security participating, and publish an anonymized after-action to
  reinforce cultural awareness.
