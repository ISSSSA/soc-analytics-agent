# Playbook: Container Escape Response

## Overview

A container escape occurs when an attacker who has code execution inside
a container breaks the isolation boundary and obtains code execution on
the host — effectively converting a workload compromise into a node
compromise. From there, the adversary can move laterally across every
workload scheduled on the same host, harvest service-account tokens
exposed by the kubelet, and potentially pivot into the cluster control
plane. Common escape paths include exploiting a vulnerable container
runtime (CVE-2019-5736 runc, CVE-2022-0185 overlayfs), abusing
over-privileged pod specs (`hostPID`, `hostNetwork`, `privileged: true`,
excess capabilities), mounted host paths, and kernel vulnerabilities
reachable from the container's syscall surface.

## Detection Indicators

- EDR or Falco alerts for unexpected syscalls from a container's cgroup:
  `mount`, `pivot_root`, `unshare`, `ptrace`, `kmod` loads.
- Privileged pods appearing in the cluster audit log that were not
  approved in the admission-policy record.
- Unexpected writes to `/var/run/docker.sock`, `/var/lib/kubelet`, or
  host paths like `/etc`, `/root` from inside a container.
- Processes with parent PID mapping to the runtime (containerd-shim,
  runc) spawning unexpected interactive shells (`sh`, `bash`, `busybox`)
  on the host namespace.
- Kube-apiserver audit events showing a new service-account token
  creation for a non-human identity during the incident window.

## MITRE ATT&CK Mapping

- **T1611** — Escape to Host. Primary technique.
- **T1068** — Exploitation for Privilege Escalation, when a kernel or
  runtime CVE is the escape vector.
- **T1552.001** — Credentials from Files. Host filesystem yields
  kubelet certs, bound service-account tokens.
- **T1078.004** — Cloud Accounts. If the node's IMDS exposes cloud IAM
  credentials, pivot to cloud-plane compromise.

## Immediate Actions (first 15 minutes)

1. Cordon the affected node (`kubectl cordon <node>`) to prevent the
   scheduler from placing new workloads on it.
2. Snapshot the node disk (cloud-provider volume snapshot or
   `dd` to external target) and collect running-process memory if
   time permits — this is the critical forensic artifact.
3. Identify and stop the offending pod but do **not** delete its
   volumes yet; the kubelet may hold state relevant for triage.
4. Isolate the node from the cluster network (Calico/Cilium policy,
   or by removing from the LB/Service pools). Do not power off
   until forensic capture is complete.
5. Page the platform on-call and the security incident commander; a
   node compromise is a cluster-level event by default.

## Investigation Steps

- Pull the offending pod's YAML spec — look for `privileged: true`,
  `hostPID: true`, `hostNetwork: true`, broad `capabilities.add`,
  `hostPath` volume mounts, or disabled `readOnlyRootFilesystem`.
- Diff the running container image against the registry manifest and
  image-signing record; unexpected image digests indicate supply-chain
  tampering.
- Review Linux audit logs (`auditd`) on the node for execve events
  outside expected binaries, and check for loaded kernel modules
  (`lsmod`) that weren't there at boot.
- Inspect the cluster audit log for API calls made with the pod's
  service-account token during the window — token reuse across
  unexpected namespaces is a red flag.
- If the node runs in a cloud VM, query the IMDS access log and the
  cloud-provider CloudTrail/Audit log for unexpected API calls made
  with the instance identity.

## Containment & Eradication

- Rotate the node's kubelet client certificate and any bootstrap tokens
  it was using; remove the node from the cluster via
  `kubectl drain --force --delete-emptydir-data` then
  `kubectl delete node`.
- Revoke all service-account tokens that were mounted on the node at
  the compromise time; for bound tokens issued by TokenRequest API,
  invalidate the issuer and re-issue.
- If a cloud IAM role was exposed via IMDS, rotate its trust policy
  and audit the last 24h of API calls made under it.
- For each container image that ran on the node, re-scan with an
  updated SBOM/CVE scanner and patch images with unpatched critical
  vulnerabilities before scheduling them elsewhere.

## Recovery

- Rebuild the node from a signed, up-to-date golden image; do not
  patch in place — a compromised node cannot be safely re-trusted.
- Restore workloads onto healthy nodes with tightened Pod Security
  Admission (restricted profile) enforced for the namespace.
- Communicate incident status, regulatory exposure, and any tenant
  data-plane impact to leadership and, where applicable, customers.

## Post-Incident

- Enforce Pod Security Admission at the `restricted` level by default
  and document exceptions with named approvers and expiry dates.
- Deploy runtime security (Falco, Tetragon, cloud-native equivalents)
  with rules tuned for escape-related syscalls and with alerts routed
  into the SIEM.
- Ban or gate `privileged`, `hostPID`, `hostNetwork`, `hostPath`, and
  the `SYS_ADMIN`, `NET_ADMIN`, `CAP_SYS_MODULE` capabilities through
  a policy engine (Kyverno, Gatekeeper, admission webhook).
- Add supply-chain controls: image signature verification (cosign),
  SBOM attestations, admission policies that block unsigned images and
  images older than N days without explicit exemption.
- Establish a quarterly "node-compromise" tabletop to keep the steps
  above muscle-memory for the platform and security on-call teams.
