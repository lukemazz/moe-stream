#!/bin/sh
# Uso: ./commit.sh "messaggio del commit" [--push]
# Crea un commit con il profilo di Luca e, con --push, lo invia su GitHub.

set -e

# >>> EDITA QUI CON I TUOI DATI <<<
GIT_NAME="lukemazz"
GIT_EMAIL="mucalazzoni@gmail.com"

if [ -z "$1" ]; then
    echo "Uso: $0 \"messaggio del commit\" [--push]" >&2
    exit 1
fi

MSG="$1"

git add -A
GIT_AUTHOR_NAME="$GIT_NAME" GIT_AUTHOR_EMAIL="$GIT_EMAIL" \
GIT_COMMITTER_NAME="$GIT_NAME" GIT_COMMITTER_EMAIL="$GIT_EMAIL" \
git commit -m "$MSG"

if [ "$2" = "--push" ]; then
    git push origin "$(git rev-parse --abbrev-ref HEAD)"
fi
