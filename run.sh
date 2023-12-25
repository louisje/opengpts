#!/usr/bin/env bash

docker image prune -f \
    && docker-compose up --build "$@"

