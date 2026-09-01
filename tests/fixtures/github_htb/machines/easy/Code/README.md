---
layout: default
title: "Code"
parent: Easy Machines
grand_parent: Machines
permalink: /machines/easy/code/
---

# Code

## Summary

Python in-browser code editor. Dangerous keywords are blocklisted, but the
execution environment still holds dangerous objects in memory.

## Key Techniques

- Python sandbox escape via subclass enumeration
- `().__class__.__base__.__subclasses__()` -> `subprocess.Popen`

## Attack Path

### 1. Recon

```bash
nmap -p- --min-rate=10000 -sV -sC code.htb
# 22  ssh
# 5000  http -> Flask: Python Playground
```
