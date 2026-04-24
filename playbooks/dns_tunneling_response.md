# Playbook: DNS Tunneling Response

## Overview

DNS tunneling is a covert-channel technique in which an adversary
encodes command-and-control traffic or exfiltrated data inside DNS
queries and responses. Because DNS is almost always permitted outbound
from internal networks (directly or via forwarders), it remains a
reliable evasion path for beaconing, small-file exfiltration, and
interactive C2 in enterprise environments. Tooling ranges from OSS
frameworks (`iodine`, `dnscat2`, `DNSCat2/powershell-rat`) through
mature malware families (e.g., DNSMessenger, OilRig's DNSpionage) to
custom implants. The playbook below assumes the SOC has detected DNS
anomalies via an IDS, NetFlow analytics, or DNS-query volume telemetry.

## Detection Indicators

- High-entropy subdomain labels (base32/base64 encoded payloads) — a
  Shannon entropy per label above ~3.5 bits is atypical for legitimate
  traffic.
- Abnormally long query names (>80 chars) or repeated queries to the
  same parent zone with monotonically increasing labels.
- Elevated TXT, NULL, or CNAME record responses to an internal host,
  particularly when the host has no legitimate need to resolve
  infrequent TLDs.
- Sustained DNS query volume from one workstation or container well
  above its baseline (e.g., >1 query/second against a single parent
  domain for >5 minutes).
- Outbound DNS to public resolvers (1.1.1.1, 8.8.8.8, or arbitrary IPs
  on UDP/53) when policy mandates internal resolvers only.
- Beaconing regularity — queries every N seconds with low jitter visible
  in the inter-arrival distribution.

## MITRE ATT&CK Mapping

- **T1071.004** — Application Layer Protocol: DNS. Primary technique.
- **T1572** — Protocol Tunneling. Generic tunneling classification.
- **T1048.003** — Exfiltration Over Unencrypted Non-C2 Protocol, when
  DNS is used to exfiltrate without an interactive C2 session.
- **T1573** — Encrypted Channel, if the payload inside DNS is
  additionally wrapped (e.g., XOR + base32).

## Immediate Actions (first 15 minutes)

1. Sinkhole the suspicious parent domain at the authoritative internal
   resolver or upstream RPZ — point it to a controlled collector so
   continuing queries are captured for analysis without reaching the
   attacker.
2. Block outbound UDP/53 and TCP/53 from non-resolver hosts at the edge
   firewall; force all DNS through the inspected internal resolver.
3. Identify the source endpoint(s) by correlating resolver logs against
   DHCP and authentication data; isolate the host at the network layer
   (disable switch port, NAC quarantine VLAN, or EDR isolation).
4. Snapshot the host's process list, network connections, loaded
   modules, and autoruns before containment changes the evidence state.
5. Engage threat intel to pivot on the parent domain — age, registrar,
   passive DNS history, and overlap with known threat-actor
   infrastructure.

## Investigation Steps

- Decode sampled query labels from the SIEM to recover the payload.
  Base32 is most common for DNS-safe encoding; check for GET/POST-like
  HTTP verbs or filenames in decoded bytes.
- Pull the host's DNS resolver cache and recent `hosts` file; persistent
  resolver overrides suggest rootkit or admin-level compromise.
- Review process ancestry around the first queried timestamp — spawning
  process, command line, parent PID — to find the implant.
- Examine endpoint for staging directories: `%TEMP%`, `%APPDATA%`,
  `/tmp`, `/dev/shm`; hash contents and triage against YARA rules.
- Check for companion indicators — scheduled tasks, cron jobs, systemd
  units, WMI subscriptions — establishing persistence.

## Containment & Eradication

- Maintain the sinkhole until threat intelligence confirms the
  C2 cluster is fully mapped; prematurely killing the channel lets the
  adversary rotate infrastructure.
- Remove the implant via authoritative uninstall (EDR action, manual
  scripted cleanup) and re-image the host if integrity cannot be
  demonstrated.
- Rotate credentials and session tokens that were resident on the host
  at time of compromise, including browser-saved passwords, SSH keys,
  kerberos tickets, and cloud CLI profiles.
- Review any lateral movement originating from the host during the
  tunneling window; pivot to broader IR plan if movement is confirmed.

## Recovery

- Restore the host from a known-good image or from backup predating
  first C2 contact; verify with file-integrity checks.
- Re-enable outbound DNS for the host only after EDR gives a clean
  baseline and endpoint DNS is again routed through the inspected
  resolver.
- Communicate status to stakeholders and, where applicable, report the
  incident to regulators per breach-notification obligations.

## Post-Incident

- Deploy continuous DNS-query entropy and volume anomaly detection at
  the resolver; alert on sustained deviations rather than instantaneous
  spikes to reduce false positives.
- Enforce strict egress DNS policy — no direct-to-public-resolver, no
  DoH to unapproved endpoints — and monitor for policy evasion
  (DoT/DoH on 443/TCP).
- Establish a formal DNS RPZ program with a weekly review cadence for
  added sinkholes and a 30-day retention of removed entries.
- Review whether initial vector (phishing, supply chain, unpatched
  CVE) produced other implants on sibling systems; hunt broadly for the
  same C2 fingerprint across the estate.
