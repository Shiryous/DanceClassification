#!/bin/bash

find . -type f -iname "*.m4a" -print0 | while IFS= read -r -d '' f; do
  wav="${f%.*}.wav"

  if [ ! -f "$wav" ]; then
    if ffmpeg -y -i "$f" -ac 1 -ar 44100 -sample_fmt s16 "$wav"; then
      echo "✔ Converted: $f"
      rm "$f"
      echo "🗑 Removed: $f"
    else
      echo "❌ Failed: $f"
    fi
  else
    echo "⚠️ WAV exists, skipping: $f"
  fi
done
# Do it multiple times, it works. Sometimes the m4a does not convert