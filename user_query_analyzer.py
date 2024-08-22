import boto3
import json
import re



bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

user_input = input("Enter your query: ")
modified_input = f'Extract the keywords from the user query related to a movie or tv show: \'{user_input}\' and return all the keywords in an array like [sci-fi,Robert,Korean] '


response = bedrock_client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps({
        "max_tokens": 512,
        "temperature": 0.5,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": modified_input}],
        }
    ]
    })
)
response_body_str = response['body'].read().decode('utf-8')
response_body = json.loads(response_body_str)

keywords_match = re.search(r'\[(.*?)\]', response_body['content'][0]['text'])

if keywords_match:
    # Convert the matched string to a list
    keywords_str = keywords_match.group(0)
    keywords_list = [keyword.strip().strip('"').strip("'") for keyword in keywords_str.split(',')]

    print("Extracted Keywords:", keywords_list)
else:
    print("No keywords found.")


search_query = ' '.join(keywords_list)

native_request = {
    "inputText": search_query
}

embedding_response = bedrock_client.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    body=json.dumps(native_request)
)

model_response = json.loads(embedding_response["body"].read())

embedding = model_response["embedding"]
input_token_count = model_response["inputTextTokenCount"]

print("\nYour input:")
print(f"Number of input tokens: {input_token_count}")
print(f"Size of the generated embedding: {len(embedding)}")
print("Embedding:")
print(embedding)