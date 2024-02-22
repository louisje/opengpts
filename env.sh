alias ollama='docker exec -it ollama ollama'
alias logs='while countdown 15 opengpts; do docker compose logs -t -f --tail=1000; done'
