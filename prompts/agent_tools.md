# Agent Tools for User Preference Summarization

This document describes the available tools for updating user preference summaries based on historical records. Each tool should be invoked as a function call, one per line, with the specified arguments.

## Tool List

### 1. add(record)
- **Purpose:** Add a new preference or information from the record to the summary.
- **Arguments:**
  - `record`: The new user record to be added.
- **Example:**
  ```
  add(User likes science fiction movies.)
  ```

### 2. delete(record)
- **Purpose:** Remove an existing preference or information from the summary that is no longer relevant, as indicated by the new record.
- **Arguments:**
  - `record`: The information to be removed from the summary.
- **Example:**
  ```
  delete(User dislikes horror movies.)
  ```

### 3. update(old_record, new_record)
- **Purpose:** Update an existing preference in the summary with new information from the record.
- **Arguments:**
  - `old_record`: The information currently in the summary.
  - `new_record`: The updated information from the new record.
- **Example:**
  ```
  update(User prefers action movies., User now prefers romantic comedies.)
  ```

### 4. keep(record)
- **Purpose:** Keep the information in the summary unchanged, as the new record confirms it is still valid.
- **Arguments:**
  - `record`: The information to keep unchanged.
- **Example:**
  ```
  keep(User enjoys watching movies on weekends.)
  ```

## Usage Notes
- Only output tool calls, one per line, with no extra explanation or summary.
- Use the most appropriate tool for each record based on its relationship to the current summary.
- Arguments should be concise and clearly reflect the user preference or information being manipulated. 