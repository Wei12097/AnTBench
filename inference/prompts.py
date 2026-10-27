"""Prompts for the main experiments and the two-round method."""

QA_PROMPT = """
# Role play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.

# Chain-of-Thought
Let’s think step by step.

# Output Control
1. First, you need to output your reasoning steps according to the question and table itself. The reasoning steps should follow the format below: [Reasoning steps for this question are as follows: 1. ... 2. ... ...]. Output steps until final answers get solved.

2. Then, you need to output the final answer. The final answer should follow the format below: [Answer Format] Final Answer: AnswerName1, AnswerName2... Ensure the final answer format is the last output line and can only be in the "Final Answer: AnswerName1, AnswerName2..." form, no other form.

3. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. Note: If the final answer has multiple decimals, retain two decimals.

4. Exception Handling (Strictly follow these priorities):
   - **Missing relevant information**: If the table lacks the necessary content or data to answer the question (i.e., the question asks for information not present in the table), you must output exactly: "Final Answer: Missing relevant information".
   - **Refuse to answer**: If the table data contains errors/contradictions, OR if the question results in multiple answers, you must output exactly: "Final Answer: Refuse to answer".

Let’s get start!
# Table:
{table}

# Question:
{question}
"""

DETECT_PROMPT = """
# Role Play
You are an expert in table analysis and data auditing. Your specific task is to verify the **intrinsic validity and internal consistency** of the table data that is referenced by the user's question.

# Scope & Focus (Crucial)
1.  **Define "Target Data" Broadly**:
    *   The "Target Data" is **NOT limited** to the specific cell containing the final answer.
    *   **Target Data** comprises the **entire set of data** required to derive or locate the answer. This includes:
        *   **Conditions/Filters**: The columns/cells checked to filter rows.
        *   **Operands**: All specific numbers involved in any calculation implied by the data structure.
        *   **Result Cells**: The pre-calculated value present in the table (e.g., a "Total" or "Average" row).
2.  **Check Internal Consistency**: You must verify the consistency between the **Target Data set** and the **Table's own logic**.
    *   **Definition**: Check if the data points (Operands and Results) conflict with each other within the table's context.
3.  **Independence from Question Text**:
    *   Do NOT judge [Wrong] if the table content differs from the *text* of the question.
    *   **Judge [Wrong] ONLY if the Target Data set contains internal contradictions.**

# Error Taxonomy & Examples
When auditing the "Target Data" (the whole set defined above), strictly look for the following specific error types based on **table-internal logic only**:

1.  **Calculation Error**:
    *   *Definition*: Arithmetic inconsistencies where derived values (e.g., Totals, Averages) contradict the operands (the individual items) in the table.
    *   *Example*: The table contains a "Total" row with value 100, but the mathematical sum of the corresponding sub-item rows (20+30+40) equals 90.

2.  **Statistics Error**:
    *   *Definition*: Discrepancies between reported aggregate indicators (rank, count, median) and the underlying data distribution.
    *   *Example*: An item is listed as "Rank: 1", but another item in the same list has a higher score/value, implying the ranking logic is flawed.

3.  **Logical Rule Conflict**:
    *   *Definition*: Mutually constrained entries violate intrinsic rules (e.g., Dates, Ranges).
    *   *Example*: A specific row has an "End Date" (2023-01-01) that is chronologically before its "Start Date" (2023-02-01).

4.  **Row/Col Head Name Error**:
    *   *Definition*: Semantic misalignment between headers and cell values in the Target Data columns.
    *   *Example*: A column header is labeled "Revenue" (implying currency), but the cell values are clearly dates (e.g., "2023-01-01").

5.  **Row/Col Duplicate**:
    *   *Definition*: Redundant rows/columns within the Target Data scope violating uniqueness.
    *   *Example*: Two distinct rows contain identical unique identifiers (e.g., "ID: 101") but contain conflicting attribute data.

6.  **General Knowledge Conflict**:
    *   *Definition*: Inconsistencies where tabular entries contradict established real-world facts, encyclopedic knowledge, or immutable entity properties external to the table context.
    *   *Example*: A row describing the entity "Paris" contains "Germany" in the "Country" column (geographical error).

# Core Task & Judgment Rules
You must evaluate the table based strictly on the following logic:

1.  **Criteria for [Wrong]**:
    *   Output **[Wrong]** IF AND ONLY IF any part of the **Target Data set** (Conditions, Operands, or Results) contains **intrinsic errors** defined in the Taxonomy.
    
2.  **Criteria for [Correct]**:
    *   Output **[Correct]** if the Target Data is mathematically and logically valid.
    *   Output **[Correct]** if the table **does not contain the necessary data** to form a complete Target Data set (e.g., missing operands or missing result cells).
    *   Output **[Correct]** for any other case that does not strictly meet the criteria for [Wrong].

# Chain-of-Thought
Let’s think step by step.

# Output Control
1.  **Reasoning & Rule Application**:
    First, you must output your reasoning steps following the format: [Reasoning steps for this question are as follows: 1. ... 2. ... ...].
    *   **Explicitly define the Target Data**: "The Target Data includes..."
    *   **Output the Mandatory Declaration** defined in Chain-of-Thought Step 1.
    *   **State the check**: "I am verifying if Sum('Price') equals 'Total'..."
    *   **Result**: State if an error type from the Taxonomy is found.

2.  **Final Judgment**:
    After explaining your reasoning, strictly output the final judgment.

3.  **Strict Format**:
    The last line of your response must strictly follow the format below. Do not add any other content after this line.
    [Answer Format] Final Answer: [Correct] / [Wrong]

Let's get started!

# Table:
{table}

# Question:
{question}
"""

EXTRACTION_PROMPT = """
# Role Play
You are an experienced **Data Logic Architect**. Your task is **NOT** to answer the question immediately. Instead, based on the provided **Table** and **Question**, you must construct a rigorous **"Execution & Verification Strategy"**.

# Chain-of-Thought
Please think according to the following steps:

1.  **Target & Mapping Analysis (Sourcing Data)**:
    *   **Intent Recognition**: Analyze the question to determine the specific value or entity requested by the user (the "Target").
    *   **Primary Mapping**: Locate the specific Row or Column headers in the table that **directly** contain the answer.
    *   **Secondary Mapping**: Identify other Rows/Columns that have a **logical, mathematical, or semantic correlation** with the target data.
        *   *Example (Math)*: Target "Revenue" -> Secondary "Price" & "Volume".
        *   *Example (Knowledge)*: Target "City" -> Secondary "Province" or "Country" (Hierarchical relationship).

2.  **Invariant Analysis (Finding Ground Truth)**:
    *   Look for **Structural Rules**, **Mathematical Relationships**, or **Factual Dependencies** that must hold true between the Primary and Secondary mappings.
    *   **Aggregation Rule**: Is there a "Total" row/column? (Rule: Sum of parts = Total).
    *   **Formula Rule**: Are there implied business formulas? (Rule: A * B = C).
    *   **Chronological/Logical Rule**: Are there hard constraints? (Rule: End Time >= Start Time).
    *   **Semantic/Factual Rule (Crucial for Entities)**: Do the entities have a real-world dependency? (Rule: The value in Col A [e.g., City] must factually belong to the category in Col B [e.g., Province] according to General Knowledge).

3.  **Formulate Dual-Track Strategy**:
    *   **Approach 1 (Direct Path)**: Describe the most direct steps to extract data based on **Primary Mapping**.
    *   **Approach 2 (Verification Path)**: Describe how to verify the target value using **Invariant Analysis** and **Secondary Mapping**.
    *   *Requirement*: 
        *   If the target is **Numerical**, calculate/derive using mathematical rules.
        *   If the target is **Textual/Entity**, verify against **General Knowledge** or **Logical Context** (e.g., "Verify if City X is actually located in Province Y").

# Output Control
1.  **Reasoning Process**: Briefly explain the mapping relationships and logical/factual rules you found.
2.  **Strategic Plan Output**:
    *   Please output strictly using the format below. Do not include extra conversational filler.
    
    [Strategic Plan]
    Target: [The specific value/entity requested by the question]
    Primary Mapping: [Directly relevant rows/columns]
    Secondary Mapping: [Related rows/columns used for derivation/verification]
    Invariant Rule: [The mathematical OR factual rule, e.g., "Row Total == Sum(Col A)" OR "City must match Province"]
    Approach 1 (Direct): [Specific instructions to extract data directly]
    Approach 2 (Verify): [Specific instructions to verify via Math Calculation OR General Knowledge Check]

# Input Data
## Table:
{table}

## Question:
{question}
"""

METHOD_DETECT_PROMPT = """
# Role Play
You are an expert Logic Validator and Consistency Checker. Your specific task is to verify the **intrinsic validity, internal consistency, and factual correctness** of the table content that is targeted by the question.

# Scope & Focus (Crucial)
1.  **Question as a Locator**: Use the question *strictly* to identify which specific rows, columns, or cells in the table need to be verified (referred to as "Target Data").
2.  **Check Consistency & Knowledge**:
    *   **Internal Consistency**: Verify mathematical/logical rules (e.g., Sums, Dates).
    *   **General Knowledge**: For entity/text data (e.g., Geography, Biology, Famous Facts), verify if the table content conflicts with **common sense or established facts**.
3.  **Do NOT compare with Question Premise**: Do not judge [Wrong] simply because the table data contradicts the text of the question. However, if the table data contradicts **World Knowledge** (Facts) or **Internal Math**, it is [Wrong].

# Task Definition
You are required to execute the **Approaches** sequentially.
You must determine the final status as:
*   **[Correct]**: If ALL approaches are executable, mathematically valid, AND factually accurate.
*   **[Wrong]**: If any inconsistency is found, you must identify the **Source of Error** (Calculation Error vs. Factual Error).

# Chain-of-Thought (CoT) Guide
Follow these steps strictly.

1.  **Analyze Approaches**: 
    *   Identify if the verification requires Math (Calculation) or Fact Checking (Knowledge).
    
2.  **Sequential Execution**: 
    *   **For EACH Approach**:
        1.  **Fetch Data**: Locate the specific values in the **Table**.
        2.  **Process**: 
            *   If **Numerical**: Perform the calculation steps.
            *   If **Textual/Entity**: Perform a **Fact Check**. Ask: "Is this relationship (e.g., [City] in [Province]) true in the real world?"
        3.  **Result**: Record the calculated value or the factual validity status.

3.  **Consistency Check & Root Cause Analysis (Crucial)**: 
    *   **Compare**: Compare Result 1 (Direct Read) vs. Result 2 (Verification).
    *   **If Results Match**: The data is consistent.
    *   **If Results Differ (CONFLICT DETECTED)**: You must perform a **Root Cause Analysis** to identify the source of the error:
        *   *Scenario A (Math Error)*: Does the table's explicit value differ from the calculated sum/product? (Source: The Table's "Total" or "Result" cell is wrong).
        *   *Scenario B (Fact Error)*: Does the table's value contradict established World Knowledge? (Source: The Table's entity content is factually incorrect).
        *   *Scenario C (Logical Error)*: Is there a contradiction in constraints (e.g., Start Date > End Date)?

4.  **Final Answer**: 
    *   Output [Correct] only if Math is correct AND Facts are correct. 
    *   Output [Wrong] if a specific error source was identified in Step 3.

# Output Control
**Reasoning Process**:
1.  **Approach 1 Execution**: Found data [X]. (Result/Value).
2.  **Approach 2 Execution**: Found data [Y, Z]. (Calculation Result OR Fact Check Result).
3.  **Discrepancy Analysis**: 
    *   Status: [Consistent / Inconsistent]
    *   (If Inconsistent): "The Conflict Source is [Math/Fact/Logic]. Specifically, the table states [A] but the verification proves [B]. Therefore, the table data in column [Col Name] is erroneous."
4.  **Final Answer**: Final verdict based on the analysis.

**Answer Format**:
Final Answer: [Correct] / [Wrong]
"""

METHOD_QA_PROMPT = """
# Role Play
You are an expert Table Analyst, Logic Executor. Your goal is to answer the specific **Question** based strictly on the provided **Table**. You must execute the **Approaches** (which may contain multiple distinct methods/paths) sequentially and verify that they lead to the same conclusion.

# Chain-of-Thought (CoT) Guide
Please follow these reasoning steps strictly to ensure accuracy and consistency:

1.  **Analyze Question & Approaches**: 
    *   Understand the Question's objective.
    *   Parse the **Approaches**. Identify if there are multiple methods (e.g., Method A, Method B) provided to solve the problem.

2.  **Sequential Execution (Iterate through Approaches)**: 
    *   **For each Approach/Method**:
        1.  Map logic to the **Table**: Identify specific headers, rows, or columns required by this specific approach.
        2.  Execute: Perform calculations or data extractions exactly as defined by this approach.
        3.  **Record**: Explicitly store the result for this method (e.g., "Result_Method_A").
        4.  If data is missing/incomplete: STOP immediately. Your Final Answer is: "Missing relevant information".

3.  **Consistency Verification**: 
    *   Compare the results from ALL executed methods.
        - **Consistency Rule**: Result_Method_A MUST equal Result_Method_B (and others).
        - **Conflict Handling**: If results differ (even slightly), or if the table contains internal contradictions preventing a unified answer, you MUST reject the answer.Your Final Answer is: "Refuse to answer".

4.  **Final Formatting**: 
    *   Format the result according to the "Output Control" rules.

# Output Control
1.  **Reasoning Output**: You must first output your reasoning process. It must show the execution of EACH approach and the comparison.
    *   Format: [Reasoning steps: 1. Executing Approach 1... Result is [A]. 2. Executing Approach 2... Result is [B]. 3. Comparing results: [A] matches/does not match [B]. 4. Conclusion...]
    
2.  **Final Answer Output**: After reasoning, output the final answer strictly in the specified format.
    *   Format: [Answer Format] Final Answer: AnswerName1, AnswerName2...
    *   **Constraint**: The answer must be the last line. Do not add explanations.
    *   **Numerics**: Retain exactly **two decimal places** (e.g., 12.34) unless it is an integer count.
    *   **Exception Handling**:
       - **Missing relevant information**: If the table lacks the necessary content or data to answer the question (i.e., the question asks for information not present in the table), you must output exactly: "Final Answer: Missing relevant information".
       - **Refuse to answer**: If the table data contains errors/contradictions, OR if the answers obtained from different problem-solving methods are **inconsistent/conflicting**., you must output exactly: "Final Answer: Refuse to answer".
"""

MAIN_PROMPTS = {'qa': QA_PROMPT, 'detect': DETECT_PROMPT}

METHOD_PROMPTS = {
    'first_round': EXTRACTION_PROMPT,
    'qa': METHOD_QA_PROMPT,
    'detect': METHOD_DETECT_PROMPT,
}
