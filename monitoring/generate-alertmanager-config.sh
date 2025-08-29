#!/bin/bash
set -e

# Load .env variables safely using a more robust method
if [ -f .env ]; then
  # Use a while loop to read the .env file line by line
  # This avoids issues with special characters and numeric values
  while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" == \#* ]]; then
      continue
    fi
    # Export the variable
    export "$line"
  done < .env
else
  echo ".env file not found!"
  exit 1
fi

TEMPLATE_FILE="monitoring/alertmanager.yml.template"
OUTPUT_FILE="monitoring/alertmanager.yml"

envsubst < "$TEMPLATE_FILE" > "$OUTPUT_FILE"

echo "✅ Config generated: $OUTPUT_FILE"
