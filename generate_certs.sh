#!/usr/bin/env bash
# Creates a private CA and a server certificate (valid for localhost / 127.0.0.1).
# Outputs: ca.crt (give to clients), server.pem + server.key (server only).
set -euo pipefail
cd "$(dirname "$0")"

openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.crt \
  -subj "/O=FedMed/CN=FedMed Root CA" -addext "basicConstraints=critical,CA:TRUE"

openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -subj "/O=FedMed/CN=localhost"
printf "subjectAltName=DNS:localhost,IP:127.0.0.1\n" > server.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.pem -days 365 -sha256 -extfile server.ext

rm -f server.csr server.ext ca.srl
echo "Certificates written to $(pwd). Never commit *.key files."
