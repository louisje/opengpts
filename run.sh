#!/usr/bin/env bash

if [ "$1" = "--no-prune" ]; then
    shift
else
    containers="$(docker ps -q -f name=redis -f name=nginx -f name=frontend -f name=backend -f name=ollama)"
    if [ -n "$containers" ]; then
        docker stop $containers
    fi

    docker container prune -f
    docker image prune -f
fi

if [ "$1" = "--build-only" ]; then
    shift
    docker compose build "$@"
    exit
fi

docker compose up --build "$@"

