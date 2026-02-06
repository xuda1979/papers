# Git SSH Setup for This Repository

## Problem

When pushing to this repository, you may encounter:

```
ERROR: Permission to xuda1979/yang-mills-mass-gap-verification.git denied to deploy key
fatal: Could not read from remote repository.
```

This happens because the system has multiple SSH keys, and Git is using the wrong one (e.g., `id_ed25519` associated with `ALPHAQUBIT` instead of this repository).

## Solution

The correct SSH key for this repository is `id_github`, created on January 13, 2026.

### Configure Git to Use the Correct Key

Run this command in the repository root:

```powershell
git config core.sshCommand "ssh -i C:/Users/Lenovo/.ssh/id_github"
```

This sets a repository-local configuration to always use the `id_github` key when pushing/pulling.

### Verify the Configuration

```powershell
git config --get core.sshCommand
```

Should output: `ssh -i C:/Users/Lenovo/.ssh/id_github`

### Test SSH Connection

```powershell
ssh -i C:/Users/Lenovo/.ssh/id_github -T git@github.com
```

Should authenticate as `xuda1979` (not `xuda1979/ALPHAQUBIT`).

## SSH Keys on This System

| Key File | Purpose |
|----------|---------|
| `id_github` | **Use this** - GitHub account `xuda1979` for this repo |
| `id_ed25519` | Other purposes |
| `id_chess` | Other purposes |
| `id_rsa_ag` | Other purposes |

## Notes

- The `core.sshCommand` config is local to this repository (stored in `.git/config`).
- If you clone the repository again, you'll need to run the config command again.
- The public key `id_github.pub` must be added to GitHub under **Settings > SSH and GPG keys** (not as a deploy key).
