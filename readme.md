# PostgreSQL + pgAdmin Setup via Docker

This project sets up a PostgreSQL server and pgAdmin using Docker Compose.  
It also demonstrates how to load a `.pgsql` file into the database and view the data using pgAdmin.

---

## Project Structure

```
JJM-WSSD_Whatsapp/
├── .env                 # Environment variables
├── docker-compose.yml   # Docker Compose config
├── jjm-ai_11082025.pgsql# SQL script to import
└── README.md            # This file
```

---

## Requirements

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- VS Code Dev Containers Tool

---

## Step 1: Setup Environment Variables

Create a `.env` file in the root directory with the following content:

```
POSTGRES_USER=admin
POSTGRES_PASSWORD=root@123
POSTGRES_DB=wssd

PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=admin
```

---

## Step 2: Docker Compose Configuration

Create a file named `docker-compose.yml` with this content:

```yaml
version: '3.8'

services:
  db:
    image: postgres:14
    container_name: pg_container
    restart: always
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin_snapshot
    restart: always
    env_file:
      - .env
    ports:
      - "8080:80"
    depends_on:
      - db

volumes:
  postgres_data:
```

---

## Step 3: Start Containers

To start everything:

```sh
docker-compose up -d
```

If you previously started the containers with different credentials and want a clean reset:

```sh
docker-compose down -v
docker-compose up -d
```

---

## Step 4: Access pgAdmin

Open your browser and go to:  
[http://localhost:8080](http://localhost:8080)

Login using:

- **Email:** admin@example.com  
- **Password:** admin

---

## Step 5: Connect to PostgreSQL in pgAdmin

After logging in to pgAdmin, register a new server using:

- **Name:** Postgres DB (or any name you prefer)

**Connection tab:**

- **Host name/address:** db
- **Port:** 5432
- **Username:** admin
- **Password:** root@123
- **Save Password:** Yes

Click **Save**.

You should now see the `wssd` database under this server.

---

## Step 6: Import .pgsql File

**Option A – Using Docker CLI:**

```sh
docker cp jjm-ai_11082025.pgsql pg_container:/jjm-ai_11082025.pgsql
docker exec -it pg_container bash
psql -U admin -d wssd -f /jjm-ai_11082025.pgsql
```

**Option B – Using pgAdmin GUI:**

1. In the pgAdmin browser, navigate to:  
   `Servers → Postgres DB → Databases → wssd`
2. Right-click on the database and open **Query Tool**
3. Copy the contents of your `.pgsql` file and paste it in
4. Click the **Execute ▶️** button

---

## Step 7: View Tables/Data in pgAdmin

1. Expand:  
   `Servers → Postgres DB → Databases → wssd → Schemas → public → Tables`
2. Right-click any table → **View/Edit Data → All Rows**

---

## Cleanup

To stop and remove containers and associated data:

```sh
docker-compose down -v
```

---

## License

MIT License — use freely for learning or development.