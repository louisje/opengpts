#!/usr/bin/env bash

containers="$(docker ps -aq)"
if [ -n "$containers" ]; then
    docker stop $containers
fi

docker container prune -f \
    && docker image prune -f \
    && docker-compose up --build "$@"

