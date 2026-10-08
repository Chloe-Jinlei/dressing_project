# Project rules

- Maintain one CLI, `uv run dressing`, and one configuration, `config.toml`.
- Add features to existing modules. Do not create numbered scripts, backup copies,
  duplicate READMEs, or alternate experiment entry points.
- All generated observations, videos, logs, benchmarks and temporary outputs go
  under `runs/<run-id>/`. Do not put generated files in the project root or assets.
- Keep assets self-contained with provenance. Import only explicitly needed files.
- Never replace the rabbit with a sloth or primitive and report dressing success.
- Both routes receive only approved observations, never simulator object truth.
- Route A emits bounded actions. Route B emits bounded model proposals; proposals
  require validation before becoming a planning model. Never execute model-written code.
- Distinguish interface tests, real API calls, physics validation and task success.
- Store secrets only in environment variables or an ignored local `.env`; never
  display credentials, include them in logs, or ask the user to paste them into chat.
- Use `uv` with the lockfile. Do not reuse the old project's environment or imports.
