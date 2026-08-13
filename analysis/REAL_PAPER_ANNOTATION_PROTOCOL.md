# Real-paper annotation protocol

The unit of annotation is one explicitly reported statistic in a defined scope.
Annotate the source label verbatim, a stable normalized variable name, statistic
type, value, units, population, subgroup, time period, arm, and page/table.

The single annotator performs two passes from the original source and records
both values. Disagreements are resolved against the cited table location; source
conflicts are retained and flagged rather than silently repaired. Derived values
(for example, counts converted to probabilities) are marked `derived`.

Primary numeric matching uses half a unit in the final reported decimal place.
Percentage/proportion conversion is allowed only when units make it explicit.
Statistic detection is matched by type and numeric tolerance; variable-name,
unit, and contextual-scope accuracy are scored separately. Empty outputs and
parse failures remain in denominators. Every false positive and false negative
receives one primary category: wrong population, subgroup/arm, time point,
negation, unit, table association, conflicting value, variable, statistic type,
numeric value, unsupported output, parsing failure, or no extraction.

