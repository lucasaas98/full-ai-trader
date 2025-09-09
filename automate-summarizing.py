import json
import os

import requests

prompt = """You are reading a file containing code with comments.
        Your job is to identify the comments that describe a task to be done.

        For each task you find, output the result as a JSON array where each object contains:
        - task
        - location

        Example output:
        [
            {
                "task": "Refactor authentication logic to remove duplicate checks",
                "location": "auth.js:42",
            },
            {
                "task": "Add unit tests for password reset feature",
                "location": "auth.test.js:10",
            }
        ]

        Now, analyze the provided file content and output the JSON only.
        """


# Define a function to summarize the contents of a file using the Ollama API
def summarize_file(path: str) -> str | None:
    with open(path, "r") as f:
        content = f.read()
        # print(f'File content: {content}')

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.1",
        "prompt": f"{prompt}\n\nFile:\n{content}",
        "stream": False,
    }

    response = requests.post(
        "http://192.168.1.133:11434/api/generate", headers=headers, json=data
    )

    if response.status_code == 200:
        try:
            # Try to parse the response as JSON
            response_data = response.json()

            # Check if response contains the expected structure
            if "response" in response_data:
                return response_data["response"]
            else:
                print(f"Unexpected response format: {response_data}")
                return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response.text}")
            return None
    else:
        print(f"Failed to generate summary for {path}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        try:
            error_data = response.json()
            print(f'Error: {error_data["error"]}')
            print(f'Error details: {error_data.get("error_details", "Not provided")}')
        except json.JSONDecodeError:
            print("Failed to parse error response as JSON")
        return None


def write_summary_to_md(summary: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(summary)


# Main script
if __name__ == "__main__":

    # Specify the directory to search for Python files
    directory = "."

    # Search for Python files in that directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                if "venv" in path:
                    continue
                print(f"\nProcessing file: {path}")

                # Summarize the contents of the file using Ollama API
                summary = summarize_file(path)

                # Check if we got a valid response from Ollama API
                if summary is not None:
                    # Write the summary to an MD file
                    result = "".join(path.split(".")[:-1])
                    md_path = "." + result + "_tasks.md"
                    write_summary_to_md(summary, f"{md_path}")
                    print(f"Tasks written to {md_path}")
                else:
                    print(f"Failed to generate tasks for {path}")
