FILE(REMOVE_RECURSE
  "CMakeFiles/buildtests"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/buildtests.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
