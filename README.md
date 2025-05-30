# Gemini agentic applications incremental design workshop

Support repo for a 2 hours 40 minutes workshop that was first given at Ahlen's GDG on February 22 2025.

https://www.youtube.com/watch?v=uIQlMSX5gx4&t=244s

The workshop is split into 5 modules. Each module is in a separate subfolder. Ideally, we will go through each module in ~ 30 minutes. Here are the topics we will cover:

- [instrumenting Gemini LLMs basics](1-instrumenting-gemini-llms-basics/instrumenting-gemini-llms-basics.md)
- [calling LLMs with tools](2-use-tools/use-tools.md)
- [the ReAct pattern](3-react-pattern/react-pattern.md)
- [the plan and execute pattern](4-plan-and-execute-pattern/plan-and-execute-pattern.md)
- [multi-agent systems](5-multi-agent-systems/multi-agent-systems.md)

... all Python scripts are to be run from the folder of each module.

## setup and pre requisites

- you need a working Python environment allowing for virtual environments (using `venv`)
- a Google account with access to Gemini
- optionally, a Google Cloud project to enable billing and therefore benefit from higher rate limits
