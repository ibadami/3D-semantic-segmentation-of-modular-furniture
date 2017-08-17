FILE(REMOVE_RECURSE
  "CMakeFiles/all_examples"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/all_examples.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
