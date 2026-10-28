# moto's standalone S3 server, for the local lite stack (docker-compose.lite.yml).
#
# An in-memory S3 emulator: objects disappear when the container stops. Fine
# for a smoke test; production uses Cloudflare R2 or Backblaze B2, which the
# app reaches through the same S3 API.
FROM python:3.12-slim

RUN pip install --no-cache-dir "moto[server]==5.2.4"

RUN useradd --system --no-create-home moto
USER moto
EXPOSE 9000
ENTRYPOINT ["moto_server", "-H", "0.0.0.0", "-p", "9000"]
