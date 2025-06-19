# TEAMS MODULE SPECIFICATION

## OVERVIEW

The `teams` module provides a streamlined interface to define and register AI agent teams. Each team is composed of three fixed roles:

- LeadAgent: Controls the workflow, manages iteration, and handles pause/resume logic.
- DevAgent: Generates solutions or artifacts based on the task.
- EvalAgent: Evaluates the DevAgent’s output and provides feedback or metrics.

This module automates:

- Construction of the team's shared State model
- Compilation of a LangGraph execution graph with fixed structure
- Integration of checkpointing and externally-controlled pausing
- Simplified API requiring only agent class definitions

---

## GOALS

- Eliminate the need for users to manually define state models or compile graphs
- Ensure consistent execution structure across all teams
- Provide a safe and extensible mechanism for pausing/resuming execution
- Allow runtime state control through external HTTP endpoints

---

## AGENT REQUIREMENTS

Each agent must define the following components:

class MyAgent:
    class InputModel(BaseModel): ...
    class OutputModel(BaseModel): ...
    class StateModel(BaseModel): ...

    def __call__(self, input: InputModel, state: StateModel) -> tuple[OutputModel, StateModel]:
        ...

ROLES OF EACH MODEL:

| Component       | Purpose                                      | Persisted? |
|----------------|----------------------------------------------|------------|
| InputModel     | Parameters required at each invocation       | No         |
| OutputModel    | Results returned from execution              | Optional   |
| StateModel     | Internal agent state (e.g., retry counter)   | Yes        |

---

## CREATING A TEAM

To define a team, simply call:

FactorTeam = create_team_class(
    team_name="FactorTeam",
    lead_cls=LeadAgent,
    dev_cls=DevAgent,
    eva_cls=EvalAgent,
)

This returns a class `FactorTeam` with:

- A dynamically generated `State` model
- A built-in `.compile()` method that returns a LangGraph
- Built-in pause support via a `should_pause` field

---

## AUTO-GENERATED STATE STRUCTURE

class FactorTeam.State(BaseModel):
    task: LeadAgent.InputModel
    should_pause: bool = False

    lead_state: LeadAgent.StateModel
    dev_state: DevAgent.StateModel
    eval_state: EvalAgent.StateModel

    dev_output: Optional[DevAgent.OutputModel]
    eval_result: Optional[EvalAgent.OutputModel]

NOTES:

- `should_pause` is a global team-level pause signal, updated externally.
- All state values are checkpointed automatically by LangGraph.

---

## EXECUTION GRAPH STRUCTURE (FIXED)

lead_node → dev_node → eval_node → lead_node

PAUSE BEHAVIOR:

- After every lead_node execution, the system checks `should_pause`.
- If True, it raises PauseExecution(state) and checkpoints the current state.

---

## EXTERNAL PAUSE CONTROL

External systems (e.g., HTTP servers or UIs) can modify `should_pause` by updating checkpointed state.

Example using FastAPI:

@app.post("/pause/factor-team")
def pause_team():
    state = checkpointer.get("FactorTeam")
    state["should_pause"] = True
    checkpointer.put("FactorTeam", state)
    return {"paused": True}

---

## EXAMPLE USAGE

class MyLead:
    class InputModel(BaseModel): ...
    class OutputModel(BaseModel): ...
    class StateModel(BaseModel): ...
    def __call__(self, input, state): ...

class MyDev: ...
class MyEval: ...

MyTeam = create_team_class("MyTeam", MyLead, MyDev, MyEval)

graph = MyTeam().compile()

state = MyTeam.State(
    task=MyLead.InputModel(...),
    lead_state=MyLead.StateModel(),
    dev_state=MyDev.StateModel(),
    eval_state=MyEval.StateModel(),
)

---

## BENEFITS

- Fixed architecture = predictable behavior
- Minimal boilerplate: no manual compile/state code
- Roles are clearly separated (Lead vs Dev vs Eval)
- Easy to checkpoint, pause, and resume
- Future extensibility (parallel devs, voting evaluators, etc.)

---

## FUTURE EXTENSIONS (OPTIONAL)

- Support for multiple DevAgents and ensemble strategies
- Configurable retry logic in LeadAgent
- Voting or weighted scoring in EvalAgent
- Dynamic injection of custom tools or strategies
- Frontend UIs auto-generated from Annotated/Field metadata

---

## SUMMARY

The `teams` module formalizes the design of 3-agent collaborative workflows in LangGraph.
By minimizing boilerplate and encapsulating pause/resume mechanics, it enables faster iteration,
safer execution, and better modularity across large agent-based systems.