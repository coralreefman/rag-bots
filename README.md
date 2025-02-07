# rag-bots

## Boilerplate

`cd boilerplate`  
`conda activate rag`
`uvicorn assistant:app --reload --host 0.0.0.0 --port 8080`  
`ngrok http http://localhost:8080`


`curl -X POST "https://home0001.ngrok.dev/chat" -u user:password -H "Content-Type: application/json" -d '{"message": "Hello, bot!"}'`