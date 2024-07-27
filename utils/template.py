summary_template = """
You are an expert in extracting and summarizing key information from text. Your task is to simplify the given text to its important content, retaining as much significant information as possible while removing irrelevant details, and return it in JSON format.\n

Please return the result in the following format: '{{"summary": "Enter the summary content here"}}'

The original text is as follows:\n{query}\n
"""


ref_template =  """
You are an expert at answering questions based on references, and you must answer the question strictly according to the content of the reference.

The reference is: {reference}

The question is: {question}

Your answer:
"""


no_ref_template = """
You are an expert encyclopedia.

Please explain in simple terms: {question}

Your answer:
"""
template = {"ref":ref_template,"no_ref":no_ref_template, "summary":summary_template}