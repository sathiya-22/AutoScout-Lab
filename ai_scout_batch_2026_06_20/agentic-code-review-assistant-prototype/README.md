Manual code reviews are a critical but often time-consuming bottleneck in software development, frequently leading to delays and potential oversight of subtle issues. This project introduces an Agentic Code Review Assistant designed to streamline and enhance the code review process.

The assistant leverages a large language model (Google Gemini) to perform an initial, automated analysis of code snippets. It identifies potential bugs, security vulnerabilities, performance inefficiencies, and style guideline violations. By providing immediate, actionable feedback and improvement suggestions, the AI frees human developers to concentrate on complex architectural decisions and nuanced logic, significantly accelerating the development cycle and improving overall code quality.

To use the assistant:
1.  **Set your API Key**: Ensure your Google Gemini API key is set as an environment variable: `export GEMINI_API_KEY="YOUR_API_KEY"`.
2.  **Install Dependencies**: Install the required Python packages: `pip install -r requirements.txt`.
3.  **Run Review**: Execute the script with the path to the code file you wish to review: `python main.py path/to/your_code.py`.
The AI-generated code review will be printed directly to your console.
