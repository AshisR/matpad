# Feature Specification: Matrix Operations Web Application

**Feature Branch**: `001-matrix-operations-app`  
**Created**: 2026-03-26  
**Updated**: 2026-04-18  
**Status**: Active  

**Input**: User description: "I want to build a browser based web application that would let user perform matrix operations on variable number of matrices. Autocomplete direction: "In the text box where a user can type in an algebraic expression involving various matrix functions, bring in support for intellisense, where we can autofill in the function that the user possibly is thinking from the first one or two letters typed. Create a dropdown of all the functions so that the user can choose one." The application should logically have three different modules. The second option to input matrices would be to specify using numpy way to specify multi row column matrices. The web interface should look modern, professional and sleek." Updated direction: "Let's augment the first module to enhance the experience layer to have a multiline textbox for the user to define define the matrix operations. We should plan to build a parsing library that parses the operations user has defined and enable the computation. I will define all the matrix operations we need to support next." Operations reference: "Treat the file Operations.md as a reference for all the matrix operations our tool needs to support. Parsing needs to recognize the functions and invoke the corresponding numerical linear algebra capabilities where available. For a function if an operator is specified we should treat that as an overloaded operator. One or more functions can be nested or combined to create new expressions and parser to honor operator precedence and grammar." Visual refinement direction: "The application should look more professional. The matrix grid should not feel overly wide, the interface should read as a polished math workspace, matrices should be visually framed, larger matrix sessions should benefit from subtle row and column guides, and scalar or vector outputs should feel as intentional as matrix results." Computation refinement direction: "Scalar and matrix multiplication must support both 2*A and A*2." Operator refinement direction: "The subtraction operator - must be supported anywhere binary operators are accepted." Layout refinement direction: "Remove the dedicated setup panel. Bring matrix definitions to the top as a compact chip bar where each matrix is shown as a named chip with editable row and column dimensions and a remove button. Add a separate operation bar directly beneath the chip bar for the expression textbox and action buttons. Users add more matrices with an Add Matrix button in the chip bar."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define matrices and operations from a compact top bar (Priority: P1)

As a user solving a linear algebra problem, I want to define my matrices directly from a compact bar at the top of the workspace, set each matrix's dimensions inline, and write my operations in a dedicated expression area so I can get to computation quickly without navigating a separate setup screen.

**Why this priority**: This is the primary path for all users. The definition bar and operation bar together replace the old setup panel with a lower-footprint interaction that keeps the matrix editors and results visible at all times.

**Independent Test**: Can be fully tested by adding matrices from the bar, changing their dimensions, entering a valid multiline operation expression, supplying matrix values, and confirming the application interprets the expression and returns the correct result.

**Acceptance Scenarios**:

1. **Given** a user opens the application, **When** the page loads, **Then** the application shows a matrix definition bar at the top with two default matrices (A and B at 2×2) and an expression input area directly below it, with no separate setup panel.
2. **Given** a user wants to add a matrix, **When** they click **Add Matrix**, **Then** the application appends a new chip with the next available letter and default 2×2 dimensions, and a matching input editor appears in the Matrix Input panel.
3. **Given** a user wants to remove a matrix, **When** they click the × button on a chip, **Then** that matrix is removed from the bar and its input editor is removed, provided at least one matrix remains.
4. **Given** a user changes the row or column count in a chip, **When** they commit the change (blur or Enter), **Then** the application resizes the matrix editor for that matrix, preserving any values that fit within the new dimensions.
5. **Given** a user has defined matrices and entered a valid multiline operation expression, **When** the user clicks Compute, **Then** the application evaluates the expression and returns the computed result.
6. **Given** a user defines an invalid or ambiguous operation, **When** the user attempts to compute, **Then** the application explains what part of the expression cannot be interpreted and preserves all entered data.
7. **Given** a user enters an expression that combines overloaded operators and function calls, **When** the application interprets the expression, **Then** it applies the documented grammar and operator precedence consistently.
8. **Given** a user begins typing a function name in the expression input, **When** one or more characters have been entered at the cursor position, **Then** the application shows a ranked dropdown of matching function names so the user can select and insert without typing the full name.
9. **Given** a user is working with larger matrices, **When** the grid is rendered, **Then** it presents the matrix as a bounded object with restrained width and subtle row guides.

---

### User Story 2 - Execute parsed operations through NumPy-backed computation (Priority: P2)

As a user who has defined matrices and an operation expression, I want the application to validate the parsed expression and invoke the correct NumPy-backed computation so I can trust the result is mathematically valid.

**Why this priority**: Parsing alone does not deliver value. The application must reliably translate parsed expressions into numerical routines and reject invalid inputs before computation.

**Independent Test**: Can be fully tested by submitting supported expressions with valid inputs, confirming the correct NumPy routine is called for each operation, and verifying that invalid inputs are rejected before execution.

**Acceptance Scenarios**:

1. **Given** a user has supplied matrices and a parsed operation expression, **When** they run the computation, **Then** the application maps each parsed node to the appropriate NumPy-backed routine and returns the result.
2. **Given** a parsed expression contains nested functions or overloaded operators, **When** execution begins, **Then** the application evaluates in parsed order and applies documented precedence rules.
3. **Given** a parsed operation requires specific input conditions, **When** the application validates before execution, **Then** it blocks invalid computations and explains the failing condition without invoking the numerical routine.
4. **Given** a supported operation returns multiple outputs, **When** computation completes, **Then** the application preserves each output and labels it according to the originating operation.
5. **Given** a numerical routine fails at runtime, **When** execution fails, **Then** the application presents a clear computation error without losing user-entered data.
6. **Given** a user enters a scalar-matrix multiplication expression such as `2*A` or `A*2`, **When** operands are otherwise valid, **Then** the application treats the scalar as a supported operand and returns the scaled matrix.

---

### User Story 3 - Review results in the dedicated output panel (Priority: P3)

As a user who has computed a result, I want a clear output panel that shows what was computed and what came back so I can interpret the result without confusion.

**Why this priority**: A clear results area completes the workspace and makes the application usable for repeat calculations, but depends on the matrix definition and entry flows above.

**Independent Test**: Can be fully tested by loading matrices, submitting a valid operation, and confirming the interface shows the interpreted expression and result clearly.

**Acceptance Scenarios**:

1. **Given** a result has been computed, **When** the user reviews the output panel, **Then** the application shows the result associated with the expression that produced it.
2. **Given** a result is a scalar or vector, **When** displayed, **Then** it uses a deliberate visual treatment consistent with matrix results rather than an unstyled text block.
3. **Given** an operation returns multiple matrix outputs (e.g. qr, eig), **When** computation completes, **Then** the application presents each returned value clearly and labels it according to the function output.
4. **Given** a user works on a desktop viewport, **When** they enter or adjust an expression, **Then** the Matrix Input and Results panels remain visible side by side so the result can be checked with minimal scrolling.
5. **Given** a user wants to focus on one part of the workflow, **When** they collapse a panel, **Then** the panel hides its body while keeping its header visible.
6. **Given** a user refreshes the page, **When** they previously collapsed or expanded a panel, **Then** the interface restores that panel state; the Capabilities panel starts collapsed by default.
7. **Given** a user is unsure which operation to use, **When** they search the Capabilities panel, **Then** the interface narrows the list to the most relevant matches using partial and fuzzy text matching.

---

### User Story 4 - Paste matrices using compact NumPy-style syntax (Priority: P4)

As a user comfortable with matrix notation, I want to paste matrices in a compact format so I can enter multi-row data faster than filling cells manually.

**Acceptance Scenarios**:

1. **Given** a user switches to Text input mode, **When** they paste valid matrix text (array literal, semicolon-separated, or plain newline-separated rows), **Then** the application parses the matrix and populates the values.
2. **Given** a user submits malformed matrix text, **When** parsing fails, **Then** the application highlights the issue and tells the user what needs to be corrected.

### Edge Cases

- What happens when a user removes all but one matrix?
- What happens when the expression textbox is empty, contains unsupported syntax, uses unary negation, or references matrices that were not defined?
- How does the system handle empty cells, non-numeric values, or incomplete matrix input?
- How does the system handle negative numeric values in both grid and text modes?
- What happens when matrix dimensions are incompatible for the selected operation?
- How does the system respond when a user submits malformed NumPy-style syntax?
- How does the system behave when multiple operations are defined and one intermediate step is invalid?
- How does the system handle expressions that mix overloaded operators and named functions?
- How does the system present results for operations that return multiple outputs?
- What happens when users nest functions deeply or use parentheses that change evaluation order?
- What happens when an operation requires properties the input does not satisfy?
- How does the system behave when a user changes matrix dimensions after values have been entered?
- How does the computation module handle runtime numerical failures?
- How does the system distinguish matrix-matrix multiplication from scalar-matrix multiplication?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a browser-based interface for performing matrix operations.
- **FR-002**: The system MUST allow users to add matrices one at a time via an **Add Matrix** button in the definition bar; each new matrix receives the next available letter identifier and default 2×2 dimensions.
- **FR-003**: The system MUST allow users to set the row and column count for each matrix individually using editable dimension inputs within each matrix chip.
- **FR-004**: The system MUST generate or resize input editors that match the dimensions set for each matrix.
- **FR-005**: The system MUST provide a multiline expression input in the operation bar where users define the matrix operations to be executed.
- **FR-006**: The system MUST interpret user-defined operation text through a reusable parsing capability before computation begins.
- **FR-007**: The system MUST provide a structured entry mode where users can input matrix values cell by cell.
- **FR-008**: The system MUST provide a compact text entry mode that accepts matrix notation in array-literal, semicolon-separated, and plain newline-separated formats.
- **FR-009**: The system MUST validate structured input, text input, and expression text before allowing computation to run.
- **FR-010**: The system MUST support the operations listed in [specs/Operations.md](../Operations.md): add, sub, eq, mult, pow, det, tr, T, ref, rref, dist, angle, dot, qr, diag, solve, inv, rank, lstsq, eig, schur, jnf, and norm, plus the boolean predicates isIdentity, isDiagonal, isSymmetric, isUpperTriangular, isOrthogonal, isOrthonormal, isIndependent.
- **FR-011**: The system MUST recognise operator symbols defined in Operations.md as overloaded equivalents of their named functions, including unary negation with `-expr`.
- **FR-012**: The system MUST support expressions composed of one or more supported functions and operators, including nested expressions and parenthesised groupings.
- **FR-013**: The system MUST honour the documented grammar and operator precedence model: `^` binds tighter than unary `-`, which binds tighter than `*`, which binds tighter than `+`/`-`, which binds tighter than `==`.
- **FR-014**: The system MUST translate each parsed operation node into the appropriate NumPy, NumPy.linalg, or SciPy numerical backend call before execution.
- **FR-015**: The system MUST execute parsed expressions according to the validated parse tree.
- **FR-016**: The computation module MUST validate operand count, matrix dimensions, numeric completeness, and supported operand types before invoking a numerical backend routine.
- **FR-017**: The computation module MUST validate operation-specific prerequisites before execution.
- **FR-018**: The system MUST support operations across a variable number of matrices whenever the expression is mathematically valid.
- **FR-018A**: The system MUST support scalar-matrix multiplication in both operand orders (`2*A` and `A*2`).
- **FR-019**: The system MUST prevent or clearly reject computations that reference undefined matrices, use unsupported syntax, or violate dimension constraints.
- **FR-020**: The system MUST trap numerical backend failures and surface them as user-readable computation errors.
- **FR-021**: The system MUST provide a Results panel where users can review computation output.
- **FR-022**: The system MUST display results in a readable matrix-oriented format that remains visible until the user changes the input or expression.
- **FR-023**: The system MUST preserve user-entered matrix data and expression text when validation or computation errors occur.
- **FR-024**: The system MUST present error messages in plain language that identify the invalid input or condition.
- **FR-025**: The system MUST present the workspace as: (1) a compact matrix definition bar at the top containing named matrix chips with editable dimensions and a remove control, followed by (2) an operation bar with the expression textarea and action buttons, followed by (3) Matrix Input and Results panels side by side, followed by (4) a Capabilities panel.
- **FR-026**: The system MUST make clear which matrices are involved in an interpreted operation and which result was produced.
- **FR-027**: The system MUST present multi-output operations in a way that preserves the relationship between each returned value and the function output label.
- **FR-027A**: On desktop-sized viewports, the Matrix Input panel and Results panel MUST be presented side by side.
- **FR-027B**: The Matrix Input, Results, and Capabilities panels MUST support collapsing and expanding.
- **FR-027C**: The Capabilities panel MUST start collapsed by default; collapsed/expanded state of all panels MUST persist across page refreshes.
- **FR-028**: Matrix editors and result displays MUST present matrices as bounded objects rather than full-width tables.
- **FR-029**: Matrix editors and result displays MUST provide subtle row guides that improve readability.
- **FR-030**: Scalar and vector outputs MUST use deliberate visual treatments consistent with matrix presentation.
- **FR-031**: The Capabilities panel MUST support searching by function name, operator symbol, or descriptive text including partial and fuzzy matches.
- **FR-032**: The expression textarea MUST provide an inline autocomplete dropdown filtered by prefix from the current word at the cursor; selecting a suggestion MUST insert the function name followed by `(` and place the cursor inside the argument position; the dropdown MUST be navigable by keyboard and dismissible.
- **FR-033**: A matrix chip's remove button MUST be disabled (or absent) when only one matrix remains in the session.

### Key Entities

- **Matrix Chip**: Represents a matrix definition in the top bar, including its letter identifier and editable row/column dimension inputs.
- **Matrix Definition**: The resolved state of a matrix chip: identifier, row count, column count, and current input mode.
- **Operation Expression**: The multiline user-authored text in the operation bar before parsing.
- **Parsed Operation**: The validated interpretation of the expression, including referenced matrices, operation order, and prerequisites.
- **Execution Step**: A validated computation step, including the targeted NumPy-backed routine, resolved operands, and preconditions.
- **Operation Catalog Entry**: A supported operation including its name, overloaded operator if any, and description.
- **Matrix Value Set**: The numeric contents of a matrix entered via grid or text mode.
- **Computation Result**: The outcome of an operation, including the result value or the validation error shown.
- **Rendered Result View**: The presentation model for matrix, vector, scalar, and multi-output results.
- **Workspace Session**: The current in-browser state including defined matrices, input modes, expression text, and visible result.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 90% of users can add two matrices, enter a valid operation expression, supply values, and complete a computation within 3 minutes on their first attempt.
- **SC-002**: 95% of valid matrix inputs submitted through either entry mode produce a result without requiring the user to retry due to interface ambiguity.
- **SC-003**: 100% of invalid operations caused by dimension mismatch or unsupported properties return an explanatory error instead of a silent failure.
- **SC-004**: 90% of users can successfully import a matrix using the compact text entry option without consulting external help.
- **SC-005**: 90% of users can correct an invalid expression after receiving feedback without resetting the whole workspace.
- **SC-006**: 95% of supported expressions combining named functions, overloaded operators, and parentheses are interpreted according to the documented grammar.
- **SC-007**: In usability review, users consistently identify the definition bar, expression input, matrix editor, and results panel without moderator clarification.
- **SC-008**: 100% of supported operations executed during acceptance testing either complete with correct output shape or return a clear validation or computation error.
- **SC-009**: In usability review, users describe matrix editors and result displays as bounded and readable rather than stretched.
- **SC-010**: 100% of scalar-matrix multiplication examples in both operand orders return the expected scaled matrix or a clear error.
- **SC-011**: Users who begin typing a function name in the expression area are offered a relevant autocomplete suggestion within one keystroke and can complete the insertion without leaving the keyboard.

## Assumptions

- The initial release targets desktop browser usage.
- The definition bar and operation bar together fulfil the role of the former setup panel, with reduced screen footprint.
- The Matrix Input and Results panels share a horizontal row on wider screens.
- Users may collapse the Matrix Input, Results, or Capabilities panels to keep the active calculation area compact.
- Panel visibility preferences survive page refreshes; the Capabilities panel is collapsed by default.
- The first release does not require user accounts, saved sessions, or collaboration features.
- The application runs locally in a browser session.
- Text mode matrix input follows NumPy array-literal conventions and also accepts semicolon-separated and plain newline-separated row formats.
- Operations.md is the current source of truth for the supported operation set and overloaded operators.
- The autocomplete dropdown is populated from the `/api/operations` response at page load.
