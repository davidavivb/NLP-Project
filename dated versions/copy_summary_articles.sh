#!/usr/bin/env bash
set -euo pipefail

SRC='/Users/david/Documents/MLDS/Sem 4/NLP/FInal Project /articles'
DST='/Users/david/Documents/MLDS/Sem 4/NLP/FInal Project /summary_articles'

mkdir -p "$DST"

for topic_dir in "$SRC"/*/; do
    topic=$(basename "$topic_dir")

    for source_dir in "$topic_dir"*/; do
        # Skip if not a directory (handles stray .json files at topic level)
        [[ -d "$source_dir" ]] || continue

        source=$(basename "$source_dir")
        count=0

        # Sort files so selection is deterministic; take first 2
        while IFS= read -r json_file && (( count < 2 )); do
            filename=$(basename "$json_file")
            dest_name="${topic}__${source}__${filename}"
            cp "$json_file" "$DST/$dest_name"
            echo "Copied: $dest_name"
            (( count++ ))
        done < <(find "$source_dir" -maxdepth 1 -name '*.json' | sort)

        if (( count == 0 )); then
            echo "WARNING: no JSON files found in $source_dir"
        fi
    done
done

echo ""
echo "Done. $(ls "$DST" | wc -l | tr -d ' ') files in $DST"
