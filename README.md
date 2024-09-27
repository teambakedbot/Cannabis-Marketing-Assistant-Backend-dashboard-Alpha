python3 -m venv myenv

# Activate your virtual environment

source myenv/bin/activate # On Windows, use `myenv\Scripts\activate`

# Run data ingestion

python app/data_ingestion.py

# Run dataset migration

python app/migrate_datasets.py
