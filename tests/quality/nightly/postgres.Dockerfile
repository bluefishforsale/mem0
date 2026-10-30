# The pgvector image production runs, plus the init script that creates the
# server's mem0_app database. Baked in rather than mounted; see docker-compose.yml.
FROM ankane/pgvector:v0.5.1
COPY server/init-db.sh /docker-entrypoint-initdb.d/init-db.sh
