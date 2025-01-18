# About
This sets up backend infrastructure to store data in a Postgres database.

# How to Use
There are 4 provided scripts (each is relatively simple if you want to run the commands yourself):

1. `up.sh` -> Start the database & associated services
2. `down.sh` -> Stop the services
3. `create_migration.sh` -> Generate a database migration after altering `schema.py`
4. `migrate.sh` -> Apply database migrations to your database instance after you or someone else creates a new migration
