# MiroFish — Full Technical Documentation

**A Simple and Universal Swarm Intelligence Engine, Predicting Anything**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Concepts](#2-core-concepts)
   - [Swarm Intelligence Prediction](#21-swarm-intelligence-prediction)
   - [Knowledge Graph (GraphRAG)](#22-knowledge-graph-graphrag)
   - [Multi-Agent Social Simulation](#23-multi-agent-social-simulation)
   - [The OASIS Engine](#24-the-oasis-engine)
3. [Architecture Overview](#3-architecture-overview)
   - [System Components](#31-system-components)
   - [Technology Stack](#32-technology-stack)
   - [Directory Structure](#33-directory-structure)
4. [The Five-Stage Workflow](#4-the-five-stage-workflow)
   - [Stage 1: Graph Building](#41-stage-1-graph-building)
   - [Stage 2: Environment Setup](#42-stage-2-environment-setup)
   - [Stage 3: Simulation](#43-stage-3-simulation)
   - [Stage 4: Report Generation](#44-stage-4-report-generation)
   - [Stage 5: Deep Interaction](#45-stage-5-deep-interaction)
5. [Agent System](#5-agent-system)
   - [Agent Creation from Entities](#51-agent-creation-from-entities)
   - [Agent Profiles](#52-agent-profiles)
   - [Agent Persona Generation](#53-agent-persona-generation)
   - [Agent Activity Configuration](#54-agent-activity-configuration)
   - [Entity Type Classification](#55-entity-type-classification)
6. [Simulation Mechanics](#6-simulation-mechanics)
   - [Round-Based Execution](#61-round-based-execution)
   - [Time Model and Activity Cycles](#62-time-model-and-activity-cycles)
   - [Platform Actions](#63-platform-actions)
   - [Dual-Platform Parallel Simulation](#64-dual-platform-parallel-simulation)
   - [Event System](#65-event-system)
   - [Platform Recommendation Config](#66-platform-recommendation-config)
   - [Graph Memory Updates](#67-graph-memory-updates)
7. [Report Agent](#7-report-agent)
   - [ReACT-Style Reasoning](#71-react-style-reasoning)
   - [Tool Suite](#72-tool-suite)
   - [Report Generation Flow](#73-report-generation-flow)
8. [Interview System](#8-interview-system)
   - [IPC Architecture](#81-ipc-architecture)
   - [Interview Modes](#82-interview-modes)
9. [Data Models and State Management](#9-data-models-and-state-management)
   - [Project Model](#91-project-model)
   - [Simulation State Machine](#92-simulation-state-machine)
   - [Runner State](#93-runner-state)
   - [Task Management](#94-task-management)
10. [API Reference](#10-api-reference)
    - [Graph API](#101-graph-api)
    - [Simulation API](#102-simulation-api)
    - [Report API](#103-report-api)
11. [Frontend](#11-frontend)
    - [Views and Routing](#111-views-and-routing)
    - [Components](#112-components)
12. [Configuration Reference](#12-configuration-reference)
    - [Environment Variables](#121-environment-variables)
    - [Application Config](#122-application-config)
13. [External Dependencies](#13-external-dependencies)
    - [Zep Cloud](#131-zep-cloud)
    - [OASIS](#132-oasis)
    - [LLM Provider](#133-llm-provider)

---

## 1. Introduction

MiroFish is a next-generation AI prediction engine powered by multi-agent swarm intelligence. Rather than using traditional statistical models or financial prediction markets, MiroFish constructs a **parallel digital world** where AI agents with independent personalities, long-term memory, and behavioral logic interact on simulated social media platforms (Twitter and Reddit). The emergent collective behavior of these agents produces prediction insights that are synthesized into structured reports.

The core loop is simple:

1. **Upload seed materials** — news articles, policy drafts, novels, financial reports, or any text documents
2. **Describe your prediction requirement** in natural language (e.g., "How will public opinion evolve around this policy?")
3. **MiroFish returns** a detailed prediction report and a fully interactive digital world you can query

Use cases span from serious decision-support (policy testing, public opinion forecasting, financial scenario analysis) to creative exploration (deducing novel endings, exploring "what if" scenarios).

---

## 2. Core Concepts

### 2.1 Swarm Intelligence Prediction

MiroFish's prediction methodology is fundamentally different from statistical forecasting or prediction markets. Instead of aggregating human bets or fitting regression models, it **simulates the social dynamics** that produce real-world outcomes.

The thesis: by creating agents whose personalities, knowledge, and behavioral patterns mirror real-world actors, and letting them freely interact in a social environment, the emergent patterns of discourse, sentiment shifts, and opinion clustering provide predictive signal about how real events might unfold.

This approach is particularly powerful for:

- **Public opinion dynamics** — how discourse evolves across communities
- **Narrative analysis** — how storylines might develop based on character psychology
- **Policy impact** — how different stakeholder groups react to proposed changes
- **Crisis simulation** — how information cascades and sentiment shifts during events

### 2.2 Knowledge Graph (GraphRAG)

MiroFish uses **Zep Cloud** to build a knowledge graph from uploaded seed documents. This graph serves two purposes:

1. **Entity extraction** — identifying the key actors, organizations, and concepts from source material, which become the basis for simulation agents
2. **Retrieval-Augmented Generation (RAG)** — providing contextual knowledge to the Report Agent during analysis

The graph construction process involves:

- **Ontology generation** — an LLM analyzes documents and produces a structured schema of entity types (e.g., Student, Professor, Organization) and edge types (e.g., "works at", "advises") with descriptions and attributes
- **Text chunking** — documents are split into chunks (default 500 characters with 50-character overlap) for ingestion
- **Graph building** — chunks are fed to Zep, which extracts entities and relationships according to the ontology
- **Entity enrichment** — entities are augmented with related edges and neighbor summaries for richer context

### 2.3 Multi-Agent Social Simulation

Each entity extracted from the knowledge graph can become a simulation agent. Agents are given:

- A **persona** — a detailed personality description generated by an LLM based on the entity's attributes, relationships, and contextual information from the graph
- An **activity configuration** — behavioral parameters like posting frequency, active hours, sentiment bias, and influence weight
- A **platform identity** — profile information for Twitter (CSV format) and/or Reddit (JSON format)

During simulation, agents act autonomously on simulated social platforms, choosing actions like creating posts, commenting, liking, following, and more. Their decisions are driven by an LLM that receives the agent's persona as a system prompt, making each agent's behavior consistent with their defined personality.

### 2.4 The OASIS Engine

The simulation runtime is powered by **[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis)** by CAMEL-AI. OASIS provides:

- Simulated Twitter and Reddit platform environments
- LLM-driven agent decision-making within each round
- Action execution and social graph management
- SQLite databases for storing posts, comments, and interactions
- Support for post-simulation agent interviews

MiroFish wraps OASIS with its own pipeline for agent generation, configuration, monitoring, and reporting.

---

## 3. Architecture Overview

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3)                         │
│   Graph Builder │ Env Setup │ Simulation │ Report │ Interaction │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP (Axios)
┌──────────────────────────────▼──────────────────────────────────┐
│                     Backend (Flask API)                          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐                │
│  │ Graph BP │  │ Simulation BP│  │ Report BP  │                │
│  └─────┬────┘  └──────┬───────┘  └─────┬──────┘                │
│        │               │                │                       │
│  ┌─────▼────────────────▼────────────────▼──────┐               │
│  │              Service Layer                    │               │
│  │  OntologyGenerator  │  SimulationManager     │               │
│  │  GraphBuilder       │  SimulationRunner      │               │
│  │  TextProcessor      │  OasisProfileGenerator │               │
│  │  ZepEntityReader    │  SimConfigGenerator    │               │
│  │  ReportAgent        │  SimulationIPC         │               │
│  │  ZepTools           │  GraphMemoryUpdater    │               │
│  └──────────────────────────────────────────────┘               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
   ┌──────▼──────┐    ┌───────▼───────┐    ┌───────▼───────┐
   │  Zep Cloud  │    │  OASIS Engine │    │  LLM Provider │
   │ (GraphRAG)  │    │  (Subprocess) │    │ (OpenAI SDK)  │
   └─────────────┘    └───────────────┘    └───────────────┘
```

### 3.2 Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vue 3, Vue Router, Axios, D3.js, Vite |
| Backend | Python 3.11–3.12, Flask, uv (package manager) |
| Simulation Engine | OASIS (CAMEL-AI) |
| Knowledge Graph | Zep Cloud (GraphRAG) |
| LLM | Any OpenAI-SDK-compatible API (default: Alibaba Qwen-plus) |
| Database | SQLite (OASIS platform data), file-based JSON (state) |
| Deployment | Docker, docker-compose |

### 3.3 Directory Structure

```
MiroAnt/
├── .env.example                          # Environment variable template
├── package.json                          # Root orchestration (concurrently)
├── Dockerfile / docker-compose.yml       # Container deployment
│
├── backend/
│   ├── run.py                            # Flask entry point (port 5001)
│   ├── requirements.txt / pyproject.toml # Python dependencies
│   │
│   ├── app/
│   │   ├── __init__.py                   # Flask app factory
│   │   ├── config.py                     # Central configuration
│   │   │
│   │   ├── api/                          # REST API blueprints
│   │   │   ├── graph.py                  # Graph/project endpoints
│   │   │   ├── simulation.py             # Simulation lifecycle endpoints
│   │   │   └── report.py                 # Report generation endpoints
│   │   │
│   │   ├── models/                       # Data models
│   │   │   ├── task.py                   # Async task tracking
│   │   │   └── project.py               # Project state persistence
│   │   │
│   │   ├── services/                     # Business logic
│   │   │   ├── ontology_generator.py     # LLM-based ontology extraction
│   │   │   ├── graph_builder.py          # Zep graph construction
│   │   │   ├── text_processor.py         # Document chunking
│   │   │   ├── zep_entity_reader.py      # Graph entity extraction
│   │   │   ├── oasis_profile_generator.py# Agent persona generation
│   │   │   ├── simulation_config_generator.py # Simulation config via LLM
│   │   │   ├── simulation_manager.py     # Simulation lifecycle state machine
│   │   │   ├── simulation_runner.py      # OASIS subprocess management
│   │   │   ├── simulation_ipc.py         # Interview IPC protocol
│   │   │   ├── zep_graph_memory_updater.py # Live graph memory sync
│   │   │   ├── zep_tools.py             # Report Agent tool implementations
│   │   │   └── report_agent.py          # ReACT report generation agent
│   │   │
│   │   └── utils/                        # Shared utilities
│   │       ├── llm_client.py             # LLM API wrapper
│   │       ├── file_parser.py            # PDF/MD/TXT extraction
│   │       ├── logger.py                 # Logging configuration
│   │       ├── retry.py                  # Retry logic
│   │       └── zep_paging.py            # Zep API pagination
│   │
│   └── scripts/                          # OASIS runner scripts
│       ├── run_twitter_simulation.py     # Twitter-only simulation
│       ├── run_reddit_simulation.py      # Reddit-only simulation
│       ├── run_parallel_simulation.py    # Dual-platform parallel
│       └── action_logger.py             # Action logging utility
│
└── frontend/
    ├── vite.config.js
    └── src/
        ├── App.vue / main.js
        ├── router/index.js               # Route definitions
        ├── api/                           # Backend API clients
        │   ├── graph.js
        │   ├── simulation.js
        │   └── report.js
        ├── views/                         # Page-level components
        │   ├── Home.vue
        │   ├── MainView.vue              # 5-step process wizard
        │   ├── SimulationView.vue
        │   ├── SimulationRunView.vue
        │   ├── ReportView.vue
        │   └── InteractionView.vue
        └── components/                    # Reusable components
            ├── Step1GraphBuild.vue
            ├── Step2EnvSetup.vue
            ├── Step3Simulation.vue
            ├── Step4Report.vue
            ├── Step5Interaction.vue
            ├── GraphPanel.vue            # D3.js graph visualization
            └── HistoryDatabase.vue
```

---

## 4. The Five-Stage Workflow

MiroFish operates through a sequential five-stage pipeline. Each stage builds on the output of the previous one.

### 4.1 Stage 1: Graph Building

**Purpose:** Extract structured knowledge from raw documents and construct a queryable knowledge graph.

**Input:** PDF, Markdown, or plain text files + a natural language simulation requirement.

**Process:**

1. **File upload and text extraction** — `FileParser` extracts text from uploaded documents (PDF via parsing, MD/TXT read directly). Supported extensions: `.pdf`, `.md`, `.txt`, `.markdown`. Maximum upload size: 50MB.

2. **Ontology generation** — `OntologyGenerator` sends the extracted text and the user's simulation requirement to the LLM, which produces:
   - **Entity types** — named categories with descriptions, attributes, and examples (e.g., `Student: A university student enrolled in programs, attributes: [name, major, year]`)
   - **Edge types** — relationship types with descriptions and valid source-target pairs (e.g., `enrolled_in: Student → University`)
   - **Analysis summary** — a brief explanation of the identified structure

3. **Text chunking** — `TextProcessor` splits the extracted text into chunks of configurable size (default 500 characters) with overlap (default 50 characters) to preserve context across chunk boundaries.

4. **Zep graph construction** — `GraphBuilderService` performs the following:
   - Creates a new graph in Zep Cloud
   - Applies the generated ontology (entity types + edge types)
   - Ingests text chunks in batches of 3
   - Waits for Zep to process episodes and extract entities/relationships
   - Fetches the resulting graph data (nodes and edges)

5. **Project state update** — the project status transitions through `CREATED → ONTOLOGY_GENERATED → GRAPH_BUILDING → GRAPH_COMPLETED`.

**Output:** A Zep knowledge graph containing typed entities and their relationships, ready for entity extraction.

### 4.2 Stage 2: Environment Setup

**Purpose:** Transform graph entities into simulation agents with rich personas and behavioral configurations.

**Input:** A completed knowledge graph + optional entity type filters.

**Process:**

1. **Entity reading and filtering** — `ZepEntityReader` queries the graph for all entities, filtering to those with custom type labels (not just the default "Entity" label). Entities can be filtered by type (e.g., only "Student" and "Professor" entities). Each entity is enriched with its related edges and neighboring nodes for context.

2. **Agent profile generation** — `OasisProfileGenerator` converts each entity into a simulation agent profile. This is the most complex step:
   - The generator classifies each entity as either an **individual** (student, professor, journalist, etc.) or a **group/organization** (university, government agency, media outlet, etc.)
   - For each entity, it builds a rich context string from the entity's attributes, graph edges, related nodes, and Zep search results
   - An LLM generates a detailed profile including: bio (~200 words), persona (~2000 words covering background, personality, social media behavior, stance, and memory), age, gender, MBTI type, country, profession, and interested topics
   - Profiles are generated in parallel (configurable concurrency, default 5) with real-time file saves
   - A rule-based fallback generates basic profiles if LLM generation fails

3. **Simulation config generation** — `SimulationConfigGenerator` uses the LLM to produce:
   - **Time configuration** — simulation duration (default 72 hours), minutes per round (default 60), activity patterns
   - **Event configuration** — hot topics, narrative directions, initial seed posts with poster type assignments
   - **Agent activity configs** — per-agent behavioral parameters (see [Section 5.4](#54-agent-activity-configuration))
   - **Platform configs** — recommendation algorithm weights for Twitter and Reddit

4. **File output** — the preparation phase writes:
   - `reddit_profiles.json` — agent profiles in Reddit format
   - `twitter_profiles.csv` — agent profiles in Twitter CSV format
   - `simulation_config.json` — complete simulation configuration

**Output:** A fully configured simulation environment ready to execute, with status transitioning from `CREATED → PREPARING → READY`.

### 4.3 Stage 3: Simulation

**Purpose:** Execute the multi-agent social simulation and capture all agent interactions.

**Input:** A prepared simulation environment + platform selection + max rounds.

**Process:**

1. **Subprocess launch** — `SimulationRunner` starts an OASIS simulation script as a subprocess:
   - `run_twitter_simulation.py` for Twitter-only
   - `run_reddit_simulation.py` for Reddit-only
   - `run_parallel_simulation.py` for both platforms simultaneously

2. **Round execution** — OASIS executes rounds sequentially. In each round:
   - Agents are activated probabilistically based on their activity level, time of day, and active hours
   - Each active agent receives their persona as a system prompt and the current social feed context
   - The LLM decides what action the agent should take (post, comment, like, etc.)
   - Actions are executed on the simulated platform
   - Results are logged to `{platform}/actions.jsonl`

3. **Real-time monitoring** — a background thread reads the action logs every 2 seconds, parsing:
   - Agent actions with metadata (round, timestamp, platform, agent, action type, result)
   - `round_end` events marking round boundaries
   - `simulation_end` events marking completion

4. **Optional graph memory sync** — if enabled, `ZepGraphMemoryManager` writes simulation actions back into the Zep knowledge graph in real-time, allowing the Report Agent to access simulation data through graph queries.

5. **State tracking** — `SimulationRunState` maintains real-time statistics including current round, actions per platform, active agent counts, and completion status.

**Output:** Complete action logs (JSONL), platform databases (SQLite), and updated run state.

### 4.4 Stage 4: Report Generation

**Purpose:** Synthesize simulation results into a structured "Future Prediction Report."

**Input:** A completed (or running) simulation + its knowledge graph.

**Process:**

1. **Planning** — the Report Agent analyzes the simulation requirement and available data to produce a report outline with numbered sections.

2. **Section-by-section writing** — for each section, the agent uses a ReACT (Reasoning-Action-Thinking) loop:
   - **Think** about what information is needed
   - **Act** by calling tools (graph search, statistics, interview)
   - **Observe** tool results
   - **Write** the section content based on gathered evidence

3. **Reflection** — after all sections are drafted, the agent reviews the complete report for coherence, accuracy, and completeness, with up to 2 reflection rounds.

4. **Output** — the final report is saved as structured Markdown with an outline, per-section content, and metadata.

**Output:** A comprehensive prediction report in Markdown format.

### 4.5 Stage 5: Deep Interaction

**Purpose:** Enable post-simulation exploration through conversational interfaces.

Two interaction modes are available:

1. **Chat with Report Agent** — a conversational interface where users can ask follow-up questions about the prediction report. The Report Agent has access to graph search tools and can retrieve additional evidence to support its answers.

2. **Interview simulation agents** — direct conversation with individual agents in the simulation. Users can ask agents about their opinions, motivations, and predictions. The interview system uses IPC to communicate with the still-running OASIS environment, so each agent responds in character according to their persona.

---

## 5. Agent System

### 5.1 Agent Creation from Entities

Agents are not manually defined — they emerge from the knowledge graph. The pipeline is:

```
Uploaded Documents
      │
      ▼
  Zep Graph (entities + relationships)
      │
      ▼
  ZepEntityReader (filter by type)
      │
      ▼
  OasisProfileGenerator (LLM personas)
      │
      ▼
  Simulation Agents (Twitter + Reddit profiles)
```

Each graph entity that passes the type filter becomes one simulation agent. The entity's attributes, relationships, and graph context are synthesized into a persona that drives the agent's behavior throughout the simulation.

### 5.2 Agent Profiles

Each agent has a structured profile (`OasisAgentProfile`) with the following fields:

| Field | Description |
|-------|-------------|
| `user_id` | Numeric identifier |
| `user_name` | Platform username (generated) |
| `name` | Display name |
| `bio` | Short biography (~200 words) |
| `persona` | Detailed personality description (~2000 words) |
| `karma` | Reddit karma score (default 0) |
| `friend_count` | Social connections count |
| `follower_count` | Platform followers |
| `statuses_count` | Historical post count |
| `age` | Agent's age |
| `gender` | Agent's gender |
| `mbti` | Myers-Briggs personality type |
| `country` | Geographic location |
| `profession` | Professional role |
| `interested_topics` | List of topic interests |
| `source_entity_uuid` | Link back to original graph entity |
| `source_entity_type` | Original entity type |

Profiles are exported in two formats:
- **Twitter** — CSV with columns matching the OASIS Twitter user schema
- **Reddit** — JSON array of profile objects

### 5.3 Agent Persona Generation

The persona is the most critical part of an agent profile. It is a ~2000-word narrative generated by the LLM that covers:

1. **Basic information** — name, age, gender, professional background
2. **Background story** — how the entity fits into the simulation's context, their history and motivations
3. **Personality traits** — MBTI type, communication style, emotional tendencies, decision-making patterns
4. **Social media behavior** — posting frequency, content style, interaction patterns, platform preferences
5. **Stance and opinion** — their position on the simulation's core topics, biases, and perspective
6. **Memory and knowledge** — what they know from the source documents, their expertise areas

The LLM receives the full entity context (attributes, edges, related nodes, Zep search results) along with the simulation requirement, and generates the persona in a structured JSON format. The system uses 3 retries with decreasing temperature (starting at the configured temperature, decreasing by 0.1 each retry) for robustness.

For **individual entities** (students, professors, journalists, etc.), the persona emphasizes personal psychology, daily life, and individual perspective.

For **group/organizational entities** (universities, government agencies, media outlets), the persona emphasizes institutional voice, official positions, and organizational behavior patterns.

### 5.4 Agent Activity Configuration

Each agent receives behavioral parameters generated by the LLM (`AgentActivityConfig`):

| Parameter | Range | Description |
|-----------|-------|-------------|
| `activity_level` | 0.0 – 1.0 | Base probability of being active in any given round |
| `posts_per_hour` | float | Average number of original posts per simulated hour |
| `comments_per_hour` | float | Average number of comments per simulated hour |
| `active_hours` | list of ints | Hours (24h format) when the agent is typically active |
| `response_delay_min` | minutes | Minimum delay before responding to content |
| `response_delay_max` | minutes | Maximum delay before responding to content |
| `sentiment_bias` | -1.0 to 1.0 | Baseline sentiment (negative = critical, positive = supportive) |
| `stance` | enum | One of: `supportive`, `opposing`, `neutral`, `observer` |
| `influence_weight` | float | Multiplier for post visibility/reach (higher = more visible) |

### 5.5 Entity Type Classification

The profile generator classifies entities into two categories for persona construction:

**Individual types:**
`student`, `alumni`, `professor`, `person`, `publicfigure`, `expert`, `faculty`, `official`, `journalist`, `activist`

**Group/Organization types:**
`university`, `governmentagency`, `organization`, `ngo`, `mediaoutlet`, `company`, `institution`, `group`, `community`

Rule-based fallback defaults differ by type:

| Entity Type | Activity | Posts/hr | Influence | Typical Active Hours |
|-------------|----------|----------|-----------|---------------------|
| University / Government | 0.2 | 0.1 | 3.0 | 9:00 – 17:00 |
| Media Outlet | 0.5 | 0.8 | 2.5 | 7:00 – 23:00 |
| Professor / Expert | 0.4 | 0.3 | 2.0 | 8:00 – 21:00 |
| Student | 0.8 | 0.6 | 0.8 | Morning + Evening |
| Alumni | 0.6 | 0.4 | 1.0 | Lunch + Evening |

---

## 6. Simulation Mechanics

### 6.1 Round-Based Execution

Simulations proceed in discrete **rounds**. Each round represents a configurable span of simulated time.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_simulation_hours` | 72 (3 days) | Total simulated time span |
| `minutes_per_round` | 60 | Simulated minutes per round |
| `max_rounds` | 10 (env override) | Maximum rounds to execute |

**Round calculation:**
```
total_rounds = total_simulation_hours × 60 ÷ minutes_per_round
```
For defaults: 72 × 60 ÷ 60 = **72 rounds** (1 round = 1 simulated hour).

**Within each round:**

1. The current simulated time is calculated based on round number and time configuration
2. Each agent's activation probability is computed from their `activity_level` multiplied by the time-of-day activity multiplier
3. Activated agents receive their persona + current social feed and the LLM selects an action
4. Actions are executed sequentially on the platform
5. A `round_end` event is logged to the action log
6. The monitor thread updates the run state

### 6.2 Time Model and Activity Cycles

The simulation models realistic daily activity patterns with time-of-day multipliers:

| Period | Hours | Multiplier | Description |
|--------|-------|------------|-------------|
| **Dead hours** | 00:00 – 05:00 | 0.05 | Near-zero activity |
| **Morning** | 06:00 – 08:00 | 0.4 | Gradual wake-up |
| **Work hours** | 09:00 – 18:00 | 0.7 | Steady daytime activity |
| **Peak hours** | 19:00 – 22:00 | 1.5 | Maximum social media activity |
| **Night** | 23:00 | 0.5 | Declining activity |

An agent's effective activation probability in a given round is:

```
P(active) = activity_level × time_multiplier × (1 if current_hour in active_hours else 0.1)
```

This creates realistic social media traffic patterns — surges during evening hours, minimal overnight activity, and steady engagement during the day.

### 6.3 Platform Actions

Agents can perform different actions depending on the platform:

**Twitter actions:**

| Action | Description |
|--------|-------------|
| `CREATE_POST` | Compose and publish an original tweet |
| `LIKE_POST` | Like an existing tweet |
| `REPOST` | Retweet content |
| `QUOTE_POST` | Quote-tweet with commentary |
| `FOLLOW` | Follow another user |
| `DO_NOTHING` | Skip this round (idle) |

**Reddit actions:**

| Action | Description |
|--------|-------------|
| `CREATE_POST` | Submit a new post to a subreddit |
| `CREATE_COMMENT` | Comment on an existing post |
| `LIKE_POST` | Upvote a post |
| `DISLIKE_POST` | Downvote a post |
| `LIKE_COMMENT` | Upvote a comment |
| `DISLIKE_COMMENT` | Downvote a comment |
| `SEARCH_POSTS` | Search for posts by keyword |
| `SEARCH_USER` | Search for a user profile |
| `TREND` | Check trending content |
| `REFRESH` | Refresh the feed |
| `FOLLOW` | Follow a user |
| `MUTE` | Mute a user |
| `DO_NOTHING` | Skip this round (idle) |

### 6.4 Dual-Platform Parallel Simulation

MiroFish supports running Twitter and Reddit simulations **simultaneously** via `run_parallel_simulation.py`. In this mode:

- Both platforms run in the same process but maintain independent round counters
- Each platform has its own action log (`twitter/actions.jsonl` and `reddit/actions.jsonl`)
- Each platform has its own SQLite database for posts, comments, and social graph
- Agents may participate on both platforms, with their persona consistent across both
- The monitor thread tracks per-platform statistics independently

Platform selection options:
- `twitter` — Twitter-only simulation
- `reddit` — Reddit-only simulation
- `parallel` — Both platforms simultaneously (default)

### 6.5 Event System

The simulation configuration includes an event system for seeding and guiding discourse:

**Initial posts** — pre-configured posts that are published at the start of the simulation to seed discussion. Each initial post specifies:
- Content text
- `poster_type` — the entity type that should post it (matched to a specific agent)
- Platform target

**Scheduled events** — time-triggered events that inject new information or stimuli at specific rounds during the simulation.

**Hot topics** — trending topics that influence agent attention and discussion themes.

**Narrative direction** — LLM-generated guidance for the overall direction of discourse evolution.

### 6.6 Platform Recommendation Config

Each platform has configurable recommendation algorithm weights that influence what content agents see:

| Parameter | Description |
|-----------|-------------|
| `recency_weight` | How much recent content is prioritized |
| `popularity_weight` | How much highly-engaged content is promoted |
| `relevance_weight` | How much content matching agent interests is surfaced |
| `viral_threshold` | Engagement threshold for content to go "viral" |
| `echo_chamber_strength` | How much the algorithm reinforces existing preferences |

### 6.7 Graph Memory Updates

When the `enable_graph_memory_update` flag is set, the `ZepGraphMemoryManager` service writes agent actions back into the Zep knowledge graph during simulation execution. This creates a feedback loop where:

1. Agent actions (posts, comments, likes) are captured in real-time
2. These are formatted as graph episodes and ingested into Zep
3. The Report Agent can then query this enriched graph for post-simulation analysis
4. Interview responses are also written back to the graph

This enables the Report Agent to have access to both the original seed knowledge and the emergent simulation data through the same graph interface.

---

## 7. Report Agent

### 7.1 ReACT-Style Reasoning

The Report Agent uses a **ReACT (Reasoning, Action, Thinking)** loop for report generation. In each step:

1. **Think** — the agent reasons about what information it needs to write the current section
2. **Act** — the agent calls one or more tools to retrieve data from the knowledge graph
3. **Observe** — tool results are incorporated into the agent's context
4. **Generate** — based on accumulated evidence, the agent writes the section content

The agent has guardrails:
- Maximum tool calls per section: configurable (default 5)
- Maximum reflection rounds: configurable (default 2)
- Temperature: configurable (default 0.5)

### 7.2 Tool Suite

The Report Agent has access to specialized tools implemented in `ZepToolsService`:

| Tool | Description |
|------|-------------|
| **InsightForge** | Deep analysis tool — performs multi-faceted graph searches combining entity exploration, relationship mapping, and semantic querying to produce comprehensive insights on a topic |
| **PanoramaSearch** | Broad-spectrum search — wide-ranging graph search that returns a panoramic view of related entities, relationships, and contextual information |
| **QuickSearch** | Targeted lookup — fast, focused graph search for specific facts, entity details, or relationship verification |
| **Interview** | Agent interview tool — allows the Report Agent to pose questions to simulation agents via the IPC system, gathering first-person perspective data |

### 7.3 Report Generation Flow

```
Simulation Requirement + Graph Data
          │
          ▼
   ┌──────────────┐
   │   Planning    │ ── Produce report outline with numbered sections
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Section 1   │ ── ReACT loop: Think → Act (tools) → Observe → Write
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Section 2   │ ── ReACT loop
   └──────┬───────┘
          │
         ...
          │
          ▼
   ┌──────────────┐
   │  Section N   │ ── ReACT loop
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Reflection  │ ── Review full report, check coherence and completeness
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │    Output     │ ── Final Markdown report
   └──────────────┘
```

Progress is tracked in real-time, allowing the frontend to display incremental section completion.

---

## 8. Interview System

### 8.1 IPC Architecture

The interview system allows post-simulation conversations with individual agents. Since the OASIS simulation runs as a separate subprocess, communication uses a **file-system-based IPC** protocol:

```
Flask Backend                    OASIS Subprocess
     │                                │
     ├── Write command to ──────►     │
     │   commands/{cmd_id}.json       │
     │                                ├── Poll commands/ directory
     │                                ├── Execute interview
     │                                ├── Write response to
     │   ◄────────────────────────────┤   responses/{cmd_id}.json
     ├── Poll responses/ directory    │
     ├── Read response                │
     └── Return to user               │
```

**Command types:**
- `INTERVIEW` — single agent interview
- `BATCH_INTERVIEW` — interview multiple agents
- `CLOSE_ENV` — gracefully shut down the simulation environment

Each command is a JSON file (`IPCCommand`) containing:
- `command_id` — unique identifier
- `command_type` — one of the above types
- `payload` — command-specific data (agent ID, prompt, platform, etc.)
- `timestamp` — when the command was issued

Responses (`IPCResponse`) contain:
- `command_id` — matching the original command
- `success` — boolean
- `result` — the agent's response text
- `error` — error message if failed

### 8.2 Interview Modes

**Single interview** (`POST /api/simulation/interview`):
- Interviews one specific agent by `agent_id`
- Requires a running simulation environment
- The prompt is optimized to prevent the agent from calling tools (text-only response)
- Configurable timeout

**Batch interview** (`POST /api/simulation/interview/batch`):
- Interviews multiple agents with potentially different questions
- Each interview is specified as `{agent_id, prompt, platform}`
- Responses are returned as an array

**All-agents interview** (`POST /api/simulation/interview/all`):
- Sends the same question to every agent in the simulation
- Useful for surveys or opinion polling across the entire agent population

**Interview history** (`POST /api/simulation/interview/history`):
- Retrieves past interview exchanges from the OASIS SQLite `trace` table
- Can be filtered by agent ID

---

## 9. Data Models and State Management

### 9.1 Project Model

Projects represent the top-level container for a prediction workflow.

**Status lifecycle:**
```
CREATED → ONTOLOGY_GENERATED → GRAPH_BUILDING → GRAPH_COMPLETED
    │                                                    │
    └──────────────── FAILED ◄───────────────────────────┘
```

**Fields:**
- `project_id` — unique identifier
- `name` — user-provided project name
- `status` — current lifecycle state
- `files` — uploaded file metadata
- `ontology` — generated entity and edge types
- `graph_id` — associated Zep graph ID
- `simulation_requirement` — natural language prediction requirement

Projects are persisted as JSON files under `uploads/projects/`.

### 9.2 Simulation State Machine

Simulations track their lifecycle through a state machine:

```
CREATED → PREPARING → READY → RUNNING → COMPLETED
    │          │         │        │
    │          │         │        └──► STOPPED
    │          │         │        └──► PAUSED
    └──────────┴─────────┴────────────► FAILED
```

**`SimulationState` fields:**

| Field | Description |
|-------|-------------|
| `simulation_id` | Unique ID (format: `sim_XXXX`) |
| `project_id` | Parent project |
| `graph_id` | Associated knowledge graph |
| `enable_twitter` | Whether Twitter platform is active |
| `enable_reddit` | Whether Reddit platform is active |
| `status` | Current state (see diagram above) |
| `entities_count` | Number of entities extracted from graph |
| `profiles_count` | Number of agent profiles generated |
| `entity_types` | Types of entities included |
| `config_generated` | Whether simulation config exists |
| `config_reasoning` | LLM's reasoning for config choices |
| `current_round` | Current simulation round |
| `twitter_status` / `reddit_status` | Per-platform status |
| `created_at` / `updated_at` | Timestamps |
| `error` | Error message if failed |

### 9.3 Runner State

During execution, `SimulationRunState` tracks real-time progress:

| Field | Description |
|-------|-------------|
| `status` | Runner status (IDLE, STARTING, RUNNING, STOPPING, STOPPED, COMPLETED, FAILED) |
| `current_round` | Overall current round |
| `total_rounds` | Total rounds to execute |
| `twitter_current_round` | Twitter-specific round counter |
| `reddit_current_round` | Reddit-specific round counter |
| `twitter_simulated_hours` | Hours simulated on Twitter |
| `reddit_simulated_hours` | Hours simulated on Reddit |
| `twitter_actions_count` | Total Twitter actions |
| `reddit_actions_count` | Total Reddit actions |
| `recent_actions` | Buffer of most recent agent actions |
| `process_pid` | PID of the OASIS subprocess |

**`AgentAction` fields:**

| Field | Description |
|-------|-------------|
| `round_num` | Which round the action occurred in |
| `timestamp` | Simulated timestamp |
| `platform` | twitter or reddit |
| `agent_id` | Acting agent's ID |
| `agent_name` | Acting agent's display name |
| `action_type` | The action performed (e.g., `CREATE_POST`) |
| `action_args` | Action-specific parameters (post content, target user, etc.) |
| `result` | Outcome of the action |
| `success` | Whether the action succeeded |

### 9.4 Task Management

Long-running operations (graph building, simulation preparation, report generation) use an async task system:

**`TaskStatus` enum:** `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`

**`Task` fields:**
- `task_id` — unique identifier
- `task_type` — operation type (e.g., "graph_build", "simulation_prepare")
- `status` — current state
- `progress` — numeric progress (0–100)
- `message` — human-readable status message
- `result` — output data on completion
- `error` — error details on failure
- `metadata` — additional context

`TaskManager` is a thread-safe singleton that allows the frontend to poll for progress on any async operation.

---

## 10. API Reference

All endpoints are prefixed with their blueprint path. The backend runs on `http://localhost:5001`.

### 10.1 Graph API

Base path: `/api/graph`

#### Project Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/project/<project_id>` | Get project details |
| `GET` | `/project/list` | List all projects |
| `DELETE` | `/project/<project_id>` | Delete a project |
| `POST` | `/project/<project_id>/reset` | Reset project to initial state |

#### Ontology and Graph Building

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ontology/generate` | Upload files and generate ontology. Multipart form: `files` (PDF/MD/TXT), `simulation_requirement` (required), `project_name`, `additional_context` |
| `POST` | `/build` | Build knowledge graph (async). Body: `project_id`, `graph_name`, `chunk_size`, `chunk_overlap`. Returns `task_id` |
| `GET` | `/task/<task_id>` | Query task status and progress |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/data/<graph_id>` | Fetch graph data (nodes + edges) for visualization |
| `DELETE` | `/delete/<graph_id>` | Delete a Zep graph |

### 10.2 Simulation API

Base path: `/api/simulation`

#### Entity Queries

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/entities/<graph_id>` | Get filtered entities. Query params: `entity_types`, `enrich` |
| `GET` | `/entities/<graph_id>/<entity_uuid>` | Get single entity detail |
| `GET` | `/entities/<graph_id>/by-type/<entity_type>` | Get entities by type |

#### Simulation Lifecycle

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/create` | Create simulation. Body: `project_id`, optional `graph_id`, `enable_twitter`, `enable_reddit` |
| `POST` | `/prepare` | Prepare simulation (async). Body: `simulation_id`, `entity_types`, `use_llm_for_profiles`, `parallel_profile_count`, `force_regenerate`. Returns `task_id` |
| `POST` | `/prepare/status` | Poll preparation progress via `task_id` or `simulation_id` |
| `GET` | `/<simulation_id>` | Get simulation state |
| `GET` | `/list` | List simulations. Query: `project_id` |
| `GET` | `/history` | List simulations with project info, round counts, report IDs |

#### Profiles and Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/<simulation_id>/profiles` | Get agent profiles. Query: `platform` |
| `GET` | `/<simulation_id>/profiles/realtime` | Real-time profiles during generation |
| `GET` | `/<simulation_id>/config` | Full simulation config |
| `GET` | `/<simulation_id>/config/realtime` | Real-time config during generation |
| `GET` | `/<simulation_id>/config/download` | Download config as JSON file |
| `GET` | `/script/<script_name>/download` | Download OASIS runner scripts |
| `POST` | `/generate-profiles` | Generate profiles independently (no simulation context) |

#### Run Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start` | Start simulation. Body: `simulation_id`, `platform` (twitter/reddit/parallel), `max_rounds`, `enable_graph_memory_update`, `force` |
| `POST` | `/stop` | Stop running simulation. Body: `simulation_id` |

#### Monitoring and Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/<simulation_id>/run-status` | Current round, progress %, action counts |
| `GET` | `/<simulation_id>/run-status/detail` | Full status with per-platform breakdown and recent actions |
| `GET` | `/<simulation_id>/actions` | Paginated action history. Query: `platform`, `agent_id`, `round`, `limit`, `offset` |
| `GET` | `/<simulation_id>/timeline` | Round-by-round summary with action breakdowns |
| `GET` | `/<simulation_id>/agent-stats` | Per-agent statistics (action counts, types, activity) |
| `GET` | `/<simulation_id>/posts` | Posts from platform SQLite database |
| `GET` | `/<simulation_id>/comments` | Comments from Reddit SQLite database |

#### Interview System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/interview` | Interview single agent. Body: `simulation_id`, `agent_id`, `prompt`, `platform`, `timeout` |
| `POST` | `/interview/batch` | Batch interview. Body: `simulation_id`, `interviews` (array of `{agent_id, prompt, platform}`) |
| `POST` | `/interview/all` | Interview all agents. Body: `simulation_id`, `prompt`, `platform` |
| `POST` | `/interview/history` | Get interview history. Body: `simulation_id`, `agent_id` |
| `POST` | `/env-status` | Check if simulation environment is alive. Body: `simulation_id` |
| `POST` | `/close-env` | Gracefully shut down simulation environment. Body: `simulation_id` |

### 10.3 Report API

Base path: `/api/report`

#### Report Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Start report generation (async). Body: `simulation_id`. Returns `task_id` |
| `POST` | `/generate/status` | Poll generation progress |
| `GET` | `/<report_id>` | Get full report (outline + content) |
| `GET` | `/by-simulation/<simulation_id>` | Get report by simulation ID |
| `GET` | `/list` | List all reports |
| `GET` | `/<report_id>/download` | Download report as Markdown file |
| `DELETE` | `/<report_id>` | Delete a report |

#### Real-time Progress

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/<report_id>/progress` | Generation progress (sections completed, current phase) |
| `GET` | `/<report_id>/sections` | Get all generated sections incrementally |
| `GET` | `/<report_id>/section/<index>` | Get a specific section by index |

#### Interaction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Chat with Report Agent (conversational, tool-augmented) |
| `GET` | `/check/<simulation_id>` | Check if a report exists for a simulation |

#### Debugging and Logging

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/<report_id>/agent-log` | Structured execution log (reasoning steps, tool calls) |
| `GET` | `/<report_id>/agent-log/stream` | Full agent log stream |
| `GET` | `/<report_id>/console-log` | Plain-text console output |
| `GET` | `/<report_id>/console-log/stream` | Full console log stream |
| `POST` | `/tools/search` | Debug: direct graph search |
| `POST` | `/tools/statistics` | Debug: graph statistics |

---

## 11. Frontend

### 11.1 Views and Routing

| Route | View | Description |
|-------|------|-------------|
| `/` | `Home.vue` | Landing page with project creation and history |
| `/process/:projectId` | `MainView.vue` | 5-step process wizard for the full workflow |
| `/simulation/:simulationId` | `SimulationView.vue` | Simulation configuration and launch |
| `/simulation/:simulationId/start` | `SimulationRunView.vue` | Real-time simulation monitoring dashboard |
| `/report/:reportId` | `ReportView.vue` | Report viewing and download |
| `/interaction/:reportId` | `InteractionView.vue` | Post-simulation chat and interviews |

### 11.2 Components

| Component | Role |
|-----------|------|
| `Step1GraphBuild` | File upload interface, ontology generation trigger, graph build progress |
| `Step2EnvSetup` | Entity type selection, simulation preparation, profile/config preview |
| `Step3Simulation` | Platform selection, round configuration, start/stop controls |
| `Step4Report` | Report generation trigger, progress tracking, Markdown rendering |
| `Step5Interaction` | Dual-panel: Report Agent chat + individual agent interviews |
| `GraphPanel` | D3.js-based interactive graph visualization (nodes, edges, zoom, pan) |
| `HistoryDatabase` | Browsable list of past simulations with status and quick access |

The frontend communicates with the backend via Axios HTTP clients defined in the `api/` directory, with separate modules for graph, simulation, and report endpoints.

---

## 12. Configuration Reference

### 12.1 Environment Variables

Set in the `.env` file at the project root:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `LLM_BASE_URL` | Yes | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM API endpoint (OpenAI SDK format) |
| `LLM_MODEL_NAME` | Yes | `qwen-plus` | Model identifier |
| `ZEP_API_KEY` | Yes | — | Zep Cloud API key for GraphRAG |
| `LLM_BOOST_API_KEY` | No | — | Optional faster LLM for acceleration |
| `LLM_BOOST_BASE_URL` | No | — | Acceleration LLM endpoint |
| `LLM_BOOST_MODEL_NAME` | No | — | Acceleration model name |

### 12.2 Application Config

Defined in `backend/app/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CONTENT_LENGTH` | 50 MB | Maximum file upload size |
| `ALLOWED_EXTENSIONS` | `{pdf, md, txt, markdown}` | Accepted file types |
| `DEFAULT_CHUNK_SIZE` | 500 | Characters per text chunk |
| `DEFAULT_CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `OASIS_DEFAULT_MAX_ROUNDS` | 10 | Default simulation rounds (env overridable) |
| `OASIS_SIMULATION_DATA_DIR` | `backend/uploads/simulations/` | Simulation data storage |
| `REPORT_AGENT_MAX_TOOL_CALLS` | 5 | Max tool invocations per report section |
| `REPORT_AGENT_MAX_REFLECTION_ROUNDS` | 2 | Max self-review iterations |
| `REPORT_AGENT_TEMPERATURE` | 0.5 | LLM temperature for report generation |

---

## 13. External Dependencies

### 13.1 Zep Cloud

[Zep](https://app.getzep.com/) provides the GraphRAG infrastructure. MiroFish uses it for:

- **Graph creation and management** — storing entities and relationships extracted from documents
- **Ontology enforcement** — applying custom entity and edge type schemas
- **Text episode ingestion** — chunked document text is added as episodes for entity extraction
- **Entity and edge queries** — reading structured data back for agent generation
- **Semantic search** — the Report Agent's tools query the graph for contextual information
- **Memory persistence** — simulation actions can be written back to the graph

A free Zep Cloud account provides sufficient quota for basic usage.

### 13.2 OASIS

[OASIS (Open Agent Social Interaction Simulations)](https://github.com/camel-ai/oasis) by CAMEL-AI is the simulation engine. It provides:

- **Simulated social platforms** — Twitter and Reddit environments with feeds, posts, comments, likes, follows
- **Agent runtime** — LLM-driven agent decision-making within each simulation round
- **SQLite storage** — platform data (posts, comments, user graphs) persisted in SQLite databases
- **Action logging** — JSONL logs of all agent actions per round
- **Interview support** — post-simulation agent conversations

OASIS runs as a Python subprocess managed by `SimulationRunner`. MiroFish communicates with it through action log files, SQLite databases, and the file-based IPC system.

### 13.3 LLM Provider

MiroFish requires an LLM provider compatible with the **OpenAI SDK format**. The LLM is used throughout the pipeline:

| Stage | LLM Usage |
|-------|-----------|
| Ontology generation | Extract entity and edge types from documents |
| Profile generation | Create detailed agent personas from entity context |
| Config generation | Produce simulation parameters (time, events, activity) |
| Simulation (OASIS) | Drive agent decision-making in each round |
| Report generation | Plan, research, and write the prediction report |
| Chat interaction | Power the Report Agent's conversational interface |

The default configuration uses Alibaba's Qwen-plus model via the Dashscope API, but any OpenAI-compatible endpoint works (OpenAI, Anthropic via proxy, local models, etc.).

Token consumption can be significant — the README recommends starting with simulations under 40 rounds. An optional "boost" LLM configuration allows using a faster/cheaper model for acceleration-eligible operations.

---

*This documentation reflects the MiroFish (MiroAnt) codebase architecture and capabilities. For setup instructions, see the [README](./README-EN.md).*
