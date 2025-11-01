# CleanMyData – Development Tasks

## (IN PROGRESS)

- [ ] Add spinning backslash loading bar to verbose mode.
- [ ] Change message `(12 columns renamed)` → `12 columns normalized successfully`.
- [ ] Adjust “Text columns cleaned (11 columns)” → `11 text columns cleaned successfully.`
      - Rephrase details as: `Whitespace stripped, spacing standardized, casing normalized.`
- [ ] Review phrasing of “No significant outliers detected.” 
      - Possibly simplify to “No outliers detected.” if detection thresholds confirm no extremes.
- [ ] Add message when no missing values are found (e.g., “No missing values detected.”).
- [ ] Add summary message when no numeric columns required format correction.
- [ ] If numeric columns < X% of total, skip outlier detection and print a message such as:
      - “Outlier analysis skipped (insufficient numeric data).”
- [ ] Log all printed summaries to `cleaning_report.txt` for large datasets (to preserve console output).
- [ ] Add light progress feedback for multi-step cleaning:
      ```
      [1/6] Removing duplicates...
      [2/6] Cleaning text columns...
      ...
      [6/6] Filling missing values...
      ```
- [ ] Add dedicated cleaning for phone numbers, email addresses, and postal addresses.

---

## (PLANNED ENHANCEMENTS)

- [ ] Integrate **Rich** for structured, colored CLI output.
- [ ] Add optional logging system with file output.
- [ ] Test across more datasets with different types of dirt.
