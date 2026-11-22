# Recruiting Agent + LangSmith Evals

A recruiting assistant agent (job lookup, building candidate profiles, scoring candidates, and
sending emails to candidates) with a LangSmith evaluation harness and a GitHub Actions workflow
that runs experiments when a PR is labeled.

## Stack

- Python `>=3.11`, managed with [uv](https://docs.astral.sh/uv/)
- LangChain / LangGraph agent (`recruiting_agent/` package)
- LangSmith for tracing + evals (`eval.py`)

---

## If you forked and cloned this repo — what you need to change

Everything below is wired to a specific LangSmith workspace/project/dataset. Swap these for your
own before running.

### 1. Local environment variables (`.env`)

Create a `.env` file in the project root. It is gitignored and loaded automatically by
`recruiting_agent/recruiting_agent.py` (via `load_dotenv()`) and referenced by `langgraph.json`
(`"env": ".env"`).

| Variable | Required | What to change |
|---|---|---|
| `LANGSMITH_API_KEY` | Yes | **Your** LangSmith API key. |
| `LANGSMITH_PROJECT` | Yes | **Your** project name (where agent traces land). |
| `DATASET_NAME` | Yes* | **Your** dataset name. Read by `eval.py` when `--dataset` is not passed. |
| `LANGSMITH_WORKSPACE_ID` | If key spans multiple workspaces | **Your** workspace ID, so datasets/experiments resolve to the right workspace. |
| `LANGSMITH_ENDPOINT` | Only if non-default | Set to your region/self-hosted URL (e.g. `https://eu.api.smith.langchain.com`). Omit for default US. |
| `LANGSMITH_TRACING` | No | Defaults to `true` (set by `recruiting_agent.py`). Set `false` to disable tracing. |
| Model provider access | Yes | The agent calls a chat model (`MODEL_NAME` in `recruiting_agent/recruiting_agent.py`). Configure whatever credentials your model routing (direct provider key or LangSmith gateway) requires. |

\* `DATASET_NAME` is required if you run `eval.py` **without** `--dataset` (see below).

Copy `.env.example` to `.env` and fill in your own values (see the table above).

### Optional: use LangSmith LLM Gateway locally

Direct OpenAI access remains the default and requires your actual `OPENAI_API_KEY`.
Gateway support uses `langchain-openai>=1.4.1` and `langchain-core>=1.5.3`; the core
minimum includes LangSmith key fallback and correct handling of an empty flag.
To use [LangSmith LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway-direct-model-access)
for both the recruiting agent and its scoring model, add these values to `.env`:

```dotenv
LANGSMITH_GATEWAY=true
LANGSMITH_API_KEY=lsv2_...
```

Replace the placeholder with your real LangSmith key. Your workspace must have an
OpenAI provider secret and access to the model in `MODEL_NAME`. No local OpenAI key
is needed for these chat calls. Optionally set `LANGSMITH_GATEWAY_API_KEY` to a
separate gateway key; it takes precedence over `LANGSMITH_API_KEY` for model calls.
Keep `LANGSMITH_API_KEY` for datasets, evaluations, and tracing.

`LANGSMITH_GATEWAY` accepts `true`, `1`, or `yes` for the default US gateway, or a
regional gateway root URL such as `https://eu.gateway.smith.langchain.com`.
Unset it or use `false`, `0`, `no`, or an empty value for direct access. Do not append
`/openai/v1`: the integration adds it. Remove `OPENAI_API_BASE` and
`OPENAI_BASE_URL` overrides when using this flag to avoid conflicting endpoint
settings. `OPENAI_API_BASE` takes precedence over the flag. `LANGSMITH_ENDPOINT`
configures the LangSmith API, not the model gateway.

The recruiting agent loads `.env` with `override=True`, so `.env` takes precedence
over shell values, including an empty value. The environment checker uses the same
precedence. Tracing is independent: `LANGSMITH_TRACING=false` does not disable the
gateway. The existing GitHub Actions workflow still uses direct OpenAI credentials;
local gateway settings do not configure CI or a remote deployment. Configure the
same variables in the process that runs the deployed agent if you use Gateway there.

### 2. Dataset names

- `eval.py` resolves the dataset as `--dataset` if given, otherwise the `DATASET_NAME`
  environment variable. If neither is set it hard-fails (there is no computed default).
- Set `DATASET_NAME` in your `.env` for local runs, or pass `--dataset "<name>"` per run.
- **Create the dataset(s) in your own LangSmith workspace** — they don't come with the repo.
  Dataset examples must have inputs shaped like `eval.py`'s `evaluation_target` expects:
  `inputs["messages"][0]["content"]`, plus optional `user_id` / `thread_id`.

### 3. Project / experiment names

- `LANGSMITH_PROJECT` (env) — where agent traces land.
- `--experiment-prefix` — names the experiment (defaults to `baseline` locally; the CI workflow
  passes `pr-<number>`).

### 4. GitHub Actions secrets

The workflow `.github/workflows/run-evals-on-label.yml` reads config from **repository Actions
secrets** (not your local `.env` — that never transfers to CI).

Add these under **Settings → Secrets and variables → Actions → Secrets → Repository secrets**.
Use *repository* secrets, not *environment* secrets: the workflow job does not declare an
`environment:`, so environment-scoped secrets would never resolve.

| Secret | Notes |
|---|---|
| `LANGSMITH_API_KEY` | Required. |
| `LANGSMITH_PROJECT` | Required for trace routing / default dataset name. |
| `DATASET_NAME` | Required. The workflow reads `${{ secrets.DATASET_NAME }}` and `eval.py` hard-fails without a dataset. |
| `OPENAI_API_KEY` | Required. The agent calls the model at import time; without it the workflow fails with `openai.OpenAIError: Missing credentials`. |
| `LANGSMITH_WORKSPACE_ID` | Add if your key spans multiple workspaces. |
| `LANGSMITH_ENDPOINT` | Only if non-default (EU / self-hosted); the line is commented out in the workflow — uncomment it and add the secret. |

Add via the web UI, or with the `gh` CLI:

```bash
gh secret set LANGSMITH_API_KEY
gh secret set LANGSMITH_PROJECT
gh secret set DATASET_NAME
gh secret set OPENAI_API_KEY
gh secret set LANGSMITH_WORKSPACE_ID   # only if key spans multiple workspaces
```

### 5. Workflow specifics to review

In `.github/workflows/run-evals-on-label.yml`:

- **Trigger label**: the job runs only when a label named `run_evals` is added to a PR
  (`if: github.event.label.name == 'run_evals'`). This label does **not** exist by default — you
  must create it in your repo (or change the name here to an existing label). The label name must
  match `run_evals` exactly (case-sensitive).

  Create it via the web UI at **Issues → Labels → New label**, or with the `gh` CLI:

  ```bash
  gh label create run_evals \
    --description "Add to a PR to run the LangSmith eval workflow against its head commit" \
    --color 1D76DB
  ```

  Suggested values:
  - **Label name**: `run_evals`
  - **Description**: `Add to a PR to run the LangSmith eval workflow against its head commit`
  - **Color**: any (e.g. `#1D76DB`) — purely cosmetic, not read by the workflow.
- **Dataset**: the workflow does **not** pass `--dataset`. It sets `DATASET_NAME` from the
  `DATASET_NAME` secret, and `eval.py` reads that env var (`--dataset` overrides it if you add the
  flag). Set the `DATASET_NAME` secret to your dataset name.
- **Experiment name**: the workflow passes `--experiment-prefix "pr-<number>"`.

---

## Get the code

Fork this repo to your own account, then clone your fork:

```bash
# Replace <your-username> with your GitHub username
git clone https://github.com/<your-username>/lca-engine.git
cd lca-engine
```

## Setup

```bash
uv sync
```

## Run an eval locally

```bash
# Uses the DATASET_NAME env var (from .env)
uv run python eval.py

# Or target a specific dataset / experiment name
uv run python eval.py --dataset "my-dataset" --experiment-prefix "local-test"
```

## Run the agent

`recruiting_agent/recruiting_agent.py` exposes
`run_agent(user_message, *, user_id=None, environment="production", thread_id=None)`
and a `recruiting_agent` graph (see `langgraph.json`).

Two scripts drive the agent over batches of example recruiter requests:

```bash
uv run python3 run.py            # email-a-candidate requests (recruiter identity via user_id)
uv run python3 run_homework.py   # scoring and skill-update requests
```

## Run evals in CI

1. Add the Actions secrets (section 4).
2. Create the `run_evals` label in your repo (section 5 — it doesn't exist by default).
3. Open a PR and add the `run_evals` label → the workflow checks out the PR head, runs `uv sync`,
   and executes `eval.py` against the configured dataset.

## Run a PR's eval manually (pre-merge)

The `run_evals` label triggers CI, but you can run the exact same experiment locally against an
open PR before merging. This mirrors the workflow: check out the PR head, sync deps, run `eval.py`.

```bash
# 1. Check out the PR branch (gh handles forks; use the PR number)
gh pr checkout <PR-number>

# 2. Sync deps (the PR may have changed them)
uv sync

# 3. Run the experiment (env vars come from your local .env)
uv run python eval.py --experiment-prefix "pr-<number>"
```

Notes:

- CI evaluates the PR **head SHA**. `gh pr checkout` lands on the branch tip, which matches as long
  as no new commits are pushed while you run. To pin it exactly: `git checkout <head-sha>`.
- `--experiment-prefix` is just a label — locally you can use e.g. `pr-<number>-local` to
  distinguish your run from the CI-generated one. Defaults to `baseline` if omitted.
- For a before/after comparison, run `eval.py` on `main` and on the PR branch with different
  prefixes, then diff the experiments in the LangSmith UI.
- Switch back when done: `git checkout main`.

---

## Files

| File | Purpose |
|---|---|
| `recruiting_agent/recruiting_agent.py` | The agent graph and `run_agent` entrypoint. |
| `recruiting_agent/data_service.py` | Data access layer. |
| `recruiting_agent/recruiting_records.py` | Records / fixtures. |
| `eval.py` | LangSmith `evaluate()` harness. |
| `run.py` | Runs the agent over example email-a-candidate requests. |
| `run_homework.py` | Runs the agent over example scoring / skill-update requests. |
| `langgraph.json` | LangGraph config (graph + `.env`). |
| `.github/workflows/run-evals-on-label.yml` | Runs evals on the `run_evals` PR label. |
