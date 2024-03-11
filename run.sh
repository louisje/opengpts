#!/usr/bin/env -S bash -x

if [ "$1" = "--no-prune" ]; then
    shift
    no_prune=yes
elif [ "$1" = "--build-only" ]; then
    shift
    docker compose build
    exit
fi

if [ -z "$no_prune" ]; then
    docker container prune -f
    docker image prune -f
fi
if [ "$1" = "--prune-only" ]; then
    exit
fi

if [ "$1" = "--all" ]; then
    shift
    docker compose up -d --build
    exit
elif [ -z "$1" ]; then
    docker compose up -d --build backend
    exit
fi

for container in "$@"; do
    docker compose build "$container"
    docker compose restart "$container"
done

