# Blind GNN experiment (second version): start here

This file is safe to share with a fresh ChatGPT or Claude conversation. Do not share the main repository, its history, scientific reports, other conversations, private evaluator directory, or another provider's submissions.

You are the candidate author. A trusted operator will give you a single `packet.json` file. Read only that file. Use a fresh conversation with memory, browsing, connectors and local file/terminal tools disabled. If those conditions cannot be met, report the limitation instead of claiming a blind run.

The packet contains the scientific task, the parent source code, the Python environment, the training/prediction interface, the compute cap and any feedback you are allowed to see. Propose one code change. Return a single JSON object matching `response_contract`, with complete source files. Do not run the host controller, inspect directories, install dependencies, request historical results or perform your own additional training. The external controller executes your code in an offline GPU container.

Your code can use the labeled training set, raw CIFs and periodic graphs available inside that container. Development structures have no labels. Do not hardcode development predictions or use external datasets or pretrained weights. Emit one finite prediction per development ID to the specified JSON path. No serialized Python objects are accepted by the evaluator.

Before your code runs, the controller performs a free format check: valid JSON, the exact top-level keys, the packet ID, the session attestation, file names, size limits and Python syntax. It runs no code and reveals no score. If the check fails, the operator may send you its error list once in the same exchange; return the complete corrected JSON object. A hypothesis longer than the stated target is truncated, not rejected. The `session.model` field is an optional self-report; the controller records the registered model itself.

The operator will deliver aggregate feedback only when permitted by the experimental arm. A submission that still fails after the repair opportunity, or whose code fails at runtime, consumes the current round. A new packet is required for the next candidate. Never reuse this conversation for another provider or an independent proposal. For the recursive arm, the next fresh packet contains the accepted source and allowed history; this is the complete state to inherit.

Keep an exported transcript when available. Token usage is optional only when the app cannot report it; enter null, never invent a count. File-exchange runs compare equal scientific compute and proposal opportunities, not guaranteed equal hidden reasoning tokens.
