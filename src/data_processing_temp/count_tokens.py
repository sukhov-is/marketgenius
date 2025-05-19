import json
import tiktoken
import sys

def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    # Note: Model versions might affect token counts slightly.
    # Using logic from OpenAI cookbook for recent models.
    # Check https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # if exact counts for older/specific versions are needed.
    if model in {
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo", "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-vision-preview",
        "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model in {
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(f"Warning: gpt-3.5-turbo model version not specified. Using defaults for gpt-3.5-turbo-0125.")
        # Fallback to a recent gpt-3.5 version
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4" in model:
        print(f"Warning: gpt-4 model version not specified. Using defaults for gpt-4-turbo.")
         # Fallback to a recent gpt-4 version
        return num_tokens_from_messages(messages, model="gpt-4-turbo")
    else:
         # Fallback for unknown models or rely on tiktoken's default for the model if possible
        try:
            encoding = tiktoken.encoding_for_model(model)
            # Basic token counting if model specific logic is unavailable
            num_tokens = 0
            for message in messages:
                # Add tokens for message structure approximation if needed
                num_tokens += 3 # Assuming a basic overhead per message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                         num_tokens += 1 # Assuming overhead for name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        except KeyError:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. """
                "See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is not None: # Ensure value is not None before encoding
                 num_tokens += len(encoding.encode(str(value))) # Encode string representation
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def count_tokens_in_jsonl(file_path):
    total_tokens = 0
    line_num = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                try:
                    data = json.loads(line)
                    if "body" in data and "messages" in data["body"]:
                        model = data["body"].get("model", "gpt-4o-mini") # Default to gpt-4o-mini if not specified
                        messages = data["body"]["messages"]
                        total_tokens += num_tokens_from_messages(messages, model)
                    else:
                        print(f"Warning: Skipping line {line_num}. Missing 'body' or 'messages' key.")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {line_num}. Invalid JSON.")
                except Exception as e:
                     print(f"Warning: Skipping line {line_num}. Error processing: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return total_tokens

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default file path if no argument is provided
        file_path = "data/external/text/batch/batch_input_news_history_part3.jsonl"
        print(f"No file path provided. Using default: {file_path}")


    total_token_count = count_tokens_in_jsonl(file_path)

    if total_token_count is not None:
        print(f"\nTotal number of tokens in {file_path}: {total_token_count}") 