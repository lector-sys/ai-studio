# AI Studio (locked)

Every file in this repository is stored encrypted (AES-256, PBKDF2 key derivation).
A fresh clone is unreadable until you unlock it with the password.

## Unlock

You need Git (with Git Bash on Windows) and OpenSSL, which comes with Git for Windows.

```bash
git clone https://github.com/lector-sys/ai-studio.git
cd ai-studio
bash tools/lock.sh unlock      # type the password when asked
```

The files are then readable and work normally. Commits and pushes are encrypted automatically.
`git log -p` and `git diff` show plain text on an unlocked clone.

## Notes

- The password is kept in this clone's `.git/config`, so treat an unlocked folder as sensitive.
- File and folder names are not encrypted, only their contents.
- Without the password the contents cannot be recovered. Keep it in a password manager.
