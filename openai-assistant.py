from openai import OpenAI
 
client = OpenAI()

instructions = ''' 
You are an expert customer service rep for the housing collective HOME0001. Use the following pieces of retrieved context to answer any question. 
If you don't find an answer just admit that you don't know. Only answer questions related to HOME0001. Talk in a casual, pragmatic tone. Avoid marketing or corporate speak at all costs. Talk like you're texting. Use three sentences maximum and keep the answer concise.",
'''
 
assistant = client.beta.assistants.create(
  name="HOME0001 Customer Assistant",
  instructions=instructions,
  model="gpt-4o",
  tools=[{"type": "file_search"}],
)

assistant_id = assistant.id
print(f"Assistant created with ID: {assistant_id}")

# Create a vector store
vector_store = client.beta.vector_stores.create(name="FAQ")
vector_store_id = vector_store.id
print(f"Vector store created with ID: {vector_store_id}")
 
# Ready the files for upload to OpenAI
file_paths = ["data/home0001qa.json"]
file_streams = [open(path, "rb") for path in file_paths]
 
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
 
# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

# Update the assistant
assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

query = "What is HOME0001?"

thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": query
    }
  ]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

message_content = messages[0].content[0].text

print(message_content.value)