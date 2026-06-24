from google import genai
from google.genai import types
from langsmith.wrappers import wrap_gemini
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

client = wrap_gemini(genai.Client(vertexai=True, project="ai-projects-500014", location="us-central1"))


@traceable(run_type="tool")
def weather_retriever():
    """Retrieve current weather information."""
    return "It is sunny today"

@traceable(name="Weather Agent")
def agent(question: str):

    tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="weather_retriever",
                description="Get current weather",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={}
                ),
            )
        ]
    )
    ]

    # First call: allow Gemini to use the tool
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(
            tools=tools
        ),
    )

    # Check whether Gemini requested a function call
    function_call = None

    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_call = part.function_call
            break

    # Execute tool if requested
    if function_call and function_call.name == "weather_retriever":

        tool_result = weather_retriever()

        # Send function result back to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                question,
                response.candidates[0].content,
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name="weather_retriever",
                            response={"result": tool_result},
                        )
                    ],
                ),
            ],
        )

    return {
        "output": response.text
    }


if __name__ == "__main__":
    result = agent("What is the weather today?")
    print(result["output"])