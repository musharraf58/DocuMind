[1mdiff --git a/.env.example b/.env.example[m
[1mindex d1a99f6..92ad2ae 100644[m
[1m--- a/.env.example[m
[1m+++ b/.env.example[m
[36m@@ -3,7 +3,7 @@[m
 [m
 # Async driver (+asyncpg) is required. Host is "localhost" when running the API[m
 # on your machine; docker-compose overrides it to the "postgres" service.[m
[31m-DATABASE_URL=postgresql+asyncpg://documind:documind@localhost:5432/documind[m
[32m+[m[32mDATABASE_URL=postgresql+asyncpg://documind:documind@localhost:5433/documind[m
 REDIS_URL=redis://localhost:6379/0[m
 [m
 # Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"[m
[1mdiff --git a/docker-compose.yml b/docker-compose.yml[m
[1mindex c1c463e..1fd62f4 100644[m
[1m--- a/docker-compose.yml[m
[1m+++ b/docker-compose.yml[m
[36m@@ -34,7 +34,7 @@[m [mservices:[m
       POSTGRES_PASSWORD: documind[m
       POSTGRES_DB: documind[m
     ports:[m
[31m-      - "5432:5432"[m
[32m+[m[32m      - "5433:5432"[m
     volumes:[m
       - postgres_data:/var/lib/postgresql/data[m
     healthcheck:[m
