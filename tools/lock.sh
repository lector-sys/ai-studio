#!/usr/bin/env bash
# Password lock for this repository: every tracked file except this script, UNLOCK.md,
# .gitattributes and .gitignore is stored in git as AES-256 ciphertext (git clean/smudge
# filter). A clone stays unreadable until it is unlocked with the password. See UNLOCK.md.
#
#   bash tools/lock.sh unlock    # asks for the password, then decrypts the working tree
#   bash tools/lock.sh status    # shows whether this clone is unlocked
#
# Needs only git and openssl (both ship with Git for Windows).
set -euo pipefail

ITER=200000
password() { git config --local --get studio.lock.password; }

encrypt() {  # stdin -> base64 ciphertext; deterministic per (path, content) so git sees no phantom changes
    local tmp salt
    tmp=$(mktemp); trap 'rm -f "$tmp"' RETURN
    cat > "$tmp"
    [ -s "$tmp" ] || return 0
    if [ "$(head -c 8 "$tmp")" = "U2FsdGVk" ]; then cat "$tmp"; return; fi   # already ciphertext
    salt=$(openssl dgst -sha256 -hmac "$1:$(password)" "$tmp" | sed 's/.*= //' | cut -c1-16)
    # OpenSSL 3 omits the "Salted__" header when -S is given, so write it ourselves; `openssl enc -d` reads the salt from it.
    {
        printf 'Salted__'
        printf "$(printf '%s' "$salt" | sed 's/../\\x&/g')"
        ENC_PASS=$(password) openssl enc -aes-256-cbc -pbkdf2 -iter "$ITER" -md sha256 -S "$salt" -pass env:ENC_PASS -in "$tmp"
    } | openssl base64
}

decrypt() {  # stdin -> plaintext; passes through anything that isn't ciphertext
    local tmp
    tmp=$(mktemp); trap 'rm -f "$tmp"' RETURN
    cat > "$tmp"
    if [ "$(head -c 8 "$tmp")" = "U2FsdGVk" ]; then
        ENC_PASS=$(password) openssl enc -d -aes-256-cbc -pbkdf2 -iter "$ITER" -md sha256 -pass env:ENC_PASS -a -in "$tmp"
    else
        cat "$tmp"
    fi
}

case "${1:-}" in
    clean) encrypt "${2:-}" ;;
    smudge) decrypt ;;
    textconv) decrypt < "$2" ;;
    unlock)
        cd "$(git rev-parse --show-toplevel)"
        if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
            echo "Commit or stash local changes first." >&2; exit 1
        fi
        read -rsp "Password: " pw; echo
        # Prove the password on one encrypted file before touching the checkout.
        probe=$(git ls-files | grep -vxE 'UNLOCK.md|tools/lock.sh|\.gitattributes|\.gitignore' | head -1)
        if ! git show "HEAD:$probe" | ENC_PASS=$pw openssl enc -d -aes-256-cbc -pbkdf2 -iter "$ITER" -md sha256 -pass env:ENC_PASS -a > /dev/null 2>&1; then
            echo "Wrong password." >&2; exit 1
        fi
        git config --local studio.lock.password "$pw"
        git config --local filter.crypt.clean 'bash tools/lock.sh clean %f'
        git config --local filter.crypt.smudge 'bash tools/lock.sh smudge'
        git config --local filter.crypt.required true
        git config --local diff.crypt.textconv 'bash tools/lock.sh textconv'
        git rm -r -q --cached . && git reset -q --hard HEAD   # re-checkout through the decrypting filter
        echo "Unlocked."
        ;;
    status)
        if git config --local --get studio.lock.password > /dev/null; then echo "Unlocked"; else echo "Locked"; fi ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac
