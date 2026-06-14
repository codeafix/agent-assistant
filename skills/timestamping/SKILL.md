---
name: timestamping
description: Append a UTC timestamp to a response.
when_to_use: The user asks what time it is, or asks for a response to be timestamped or dated.
---

# Timestamping responses

When the user asks for the current time, or asks you to timestamp your
answer:

1. Call the `clock` tool (from the echo-clock MCP server) to get the current
   UTC time. Do not guess or compute it yourself.
2. Format the result as `YYYY-MM-DD HH:MM UTC` in your response.

Example response: "It's currently 2026-06-14 09:05 UTC."
