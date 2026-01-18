from mcp.server.fastmcp import FastMCP
import questions

mcp = FastMCP("questions_server")


@mcp.tool()
async def get_questions_with_answer() -> str:
    """
    Retrieve from the database all the recorded questions where you have been provided with an official answer.

    Returns:
        A string containing the questions with their official answers.
    """

    return questions.get_questions_with_answer()


@mcp.tool()
async def get_questions_with_no_answer() -> str:
    """
    Retrieve from the database all the recorded question where there is no official answer.

    Returns:
        A string containing the questions with no official answers.
    """

    return questions.get_questions_with_no_answer()


@mcp.tool()
async def record_question_with_no_answer(question: str) -> str:
    """
    Record a question to the database that currently has no answer.

    Args:
        question: The string content of the question.
    """

    return questions.record_question_with_no_answer(question)


@mcp.tool()
async def record_answer_to_question(id: int, answer: str) -> str:
    """
    Update a question in the database with an official answer.

    Args:
        id: The integer ID of the question.
        answer: The string content of the answer.
    """

    return questions.record_answer_to_question(id, answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
