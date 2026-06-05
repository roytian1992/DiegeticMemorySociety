# Wandering Earth 2 External Reference Test Corpus

- Created at: 2026-06-05
- Purpose: mixed-format external reference corpus for testing reference-library ingestion and item extraction.
- Copyright policy: this corpus stores source-grounded notes and short evidence snippets, not mirrored full webpages.
- Suggested next step: run the future `ingest-reference-library` / `extract-reference-items` pipeline on this directory, then import the generated `reference_items.jsonl`.

## Files

- `world_bible_mixed.md`: markdown notes mixing world bible, technology, location, timeline, and style guidance.
- `character_profiles.txt`: plain-text role/profile notes.
- `timeline_and_world_refs.json`: JSON document with nested timeline/world/location/style sections.
- `mixed_reference_notes.jsonl`: JSONL notes, one mixed raw note per line.
- `source_manifest.json`: source URLs and what each source was used for.

## Source URLs

- https://en.wikipedia.org/wiki/The_Wandering_Earth_2
- https://zh.wikipedia.org/wiki/%E6%B5%81%E6%B5%AA%E5%9C%B0%E7%90%832
- https://zh.wikipedia.org/wiki/%E6%B5%81%E6%B5%AA%E5%9C%B0%E7%90%83%E7%B3%BB%E5%88%97%E7%94%B5%E5%BD%B1
- https://www.gcores.com/articles/162106
- https://sino-cinema.com/2023/02/17/review-the-wandering-earth-ii-2023/
