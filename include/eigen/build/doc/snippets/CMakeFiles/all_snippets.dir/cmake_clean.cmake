FILE(REMOVE_RECURSE
  "CMakeFiles/all_snippets"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/all_snippets.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
