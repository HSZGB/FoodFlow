# PPT Workflow

FoodFlow uses the `codex-ppt` skill, whose workflow has mandatory approval gates.

Current status:

- [x] Source reading and asset extraction.
- [x] Draft outline written to `ppt/FoodFlow/outline.md`.
- [ ] User approval of outline.
- [ ] Visual style confirmation.
- [ ] Image backend confirmation.
- [ ] One sample slide approval.
- [ ] Full slide generation.
- [ ] QA, speaker notes, and PPT assembly.

No final `deck_spec.json`, `speech.md`, prompt job files, slide images, or `.pptx` should be created before the corresponding approvals.

NotebookLM fallback:

- Because the current environment cannot reliably generate slide images, `ppt/notebooklm/` provides a fallback upload pack.
- The fallback pack follows the same 11-page outline and focuses the deck on five offline strategies and four fulfillment simulation chains.
- If a future image-based PPT is generated with `codex-ppt`, restart from the approval gates above.
- If a future generated image/PDF deck needs object-level editability, use `image-to-editable-ppt`; it is a reconstruction workflow, not the authoring workflow for this project.
