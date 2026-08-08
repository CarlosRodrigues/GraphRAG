FROM postgres:16

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     postgresql-server-dev-16     git     bison     flex     ca-certificates     && rm -rf /var/lib/apt/lists/*

# Install pgvector (v0.7.4)
RUN git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git /tmp/pgvector     && make -C /tmp/pgvector     && make -C /tmp/pgvector install     && rm -rf /tmp/pgvector

# Install Apache AGE (PG16 compatible release)
RUN git clone --branch PG16/v1.5.0-rc0 https://github.com/apache/age.git /tmp/age     && make -C /tmp/age PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install     && rm -rf /tmp/age

# Copy initialization script
COPY init.sql /docker-entrypoint-initdb.d/01-init.sql