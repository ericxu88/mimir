# mimir (WIP)

in Norse mythology, Mimir guarded the well of wisdom, Mímisbrunnr, beneath the world tree. Odin gave up an eye for a single drink from it.

real knowledge comes at a cost.

same is true with llm experimentation where you can't just eyeball a few outputs and know which prompt is better

mimir is a small harness for running LLM experiments properly. You describe an experiment in a YAML file (a few prompt variants, a dataset, how many samples). mimir runs it and tells you whether the difference between variants is statistically real or just noise. mimir also checks whether the LLM judge scoring your outputs is actually reliable and robust
