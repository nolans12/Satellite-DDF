# What is this?
Alembic generates and performs database migrations when the schema in `backend/schema.py` changes.

# How do use?
- Call `./backend/create_migration.sh <migration message>` after adjusting `schema.py`.
- Call `./backend/migrate.sh` to apply migrations. This is also needed if someone else creates a migration and, after pulling the latest changes, it needs to be applied to your DB instance.
