"""
Graph retrieval tools service.

Wraps graph search, node reading, edge querying, and other tools for the Report Agent.

Core retrieval tools (optimized):
1. InsightForge (Deep insight retrieval) - Most powerful hybrid retrieval
2. PanoramaSearch (Panoramic search) - Get the full picture, including expired content
3. QuickSearch (Simple search) - Quick retrieval
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.graphiti_client import get_graphiti_sync, run_async

logger = get_logger('mirofish.graph_tools')


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        """Convert to text format for LLM understanding"""
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} related items"]

        if self.facts:
            text_parts.append("\n### Related facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }

    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node", "EntityNode", "EpisodicNode"]), "Unknown type")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge information"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "Present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """Deep insight retrieval result (InsightForge)"""
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding"""
        text_parts = [
            f"## In-depth future prediction analysis",
            f"Analysis question: {self.query}",
            f"Prediction scenario: {self.simulation_requirement}",
            f"\n### Prediction data statistics",
            f"- Related prediction facts: {self.total_facts}",
            f"- Entities involved: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}"
        ]

        if self.sub_queries:
            text_parts.append(f"\n### Analyzed sub-questions")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        if self.semantic_facts:
            text_parts.append(f"\n### Key facts")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.entity_insights:
            text_parts.append(f"\n### Core entities")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f'  Summary: "{entity.get("summary")}"')
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))}")

        if self.relationship_chains:
            text_parts.append(f"\n### Relationship chains")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """Panoramic search result"""
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        """Convert to text format (full version, no truncation)"""
        text_parts = [
            f"## Panoramic search result",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Currently active facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}"
        ]

        if self.active_facts:
            text_parts.append(f"\n### Currently active facts")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.historical_facts:
            text_parts.append(f"\n### Historical/expired facts")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.all_nodes:
            text_parts.append(f"\n### Entities involved")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node", "EntityNode", "EpisodicNode"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Interview result for a single Agent"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key quotes:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """Interview result containing responses from multiple simulated Agents."""
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        """Convert to detailed text format"""
        text_parts = [
            "## In-depth interview report",
            f"**Interview topic:** {self.interview_topic}",
            f"**Interviewed count:** {self.interviewed_count} / {self.total_agents} simulated Agents",
            "\n### Reasoning for interviewee selection",
            self.selection_reasoning or "(Auto-selected)",
            "\n---",
            "\n### Interview transcripts",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview summary and key viewpoints")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


class GraphToolsService:
    """
    Graph retrieval tools service.

    Same interface as the old ZepToolsService.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.graphiti = get_graphiti_sync()
        self._llm_client = llm_client
        logger.info("GraphToolsService initialization complete")

    @property
    def llm(self) -> LLMClient:
        """Lazily initialize LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """API call with retry mechanism"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "%s attempt %d failed: %s, %.1fs before retry...",
                        operation_name, attempt + 1, str(e)[:100], delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        "%s failed after %d attempts: %s",
                        operation_name, max_retries, str(e),
                    )

        raise last_exception

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """
        Graph semantic search.

        Uses Graphiti's hybrid search (semantic + keyword + graph traversal).
        Falls back to local keyword matching if Graphiti search fails.
        """
        logger.info("Graph search: graph_id=%s, query=%s...", graph_id, query[:50])

        try:
            edges = self._call_with_retry(
                func=lambda: run_async(
                    self.graphiti.search(
                        query=query,
                        group_ids=[graph_id],
                        num_results=limit,
                    )
                ),
                operation_name=f"Graph search (graph={graph_id})",
            )

            facts = []
            edges_data = []

            for edge in edges:
                fact = getattr(edge, "fact", "")
                if fact:
                    facts.append(fact)
                edges_data.append({
                    "uuid": getattr(edge, "uuid", ""),
                    "name": getattr(edge, "name", ""),
                    "fact": fact,
                    "source_node_uuid": getattr(edge, "source_node_uuid", ""),
                    "target_node_uuid": getattr(edge, "target_node_uuid", ""),
                })

            logger.info("Search complete: found %d related facts", len(facts))

            return SearchResult(
                facts=facts,
                edges=edges_data,
                nodes=[],
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.warning("Graphiti search failed; falling back to local search: %s", str(e))
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """Local keyword matching search (fallback)."""
        logger.info("Using local search: query=%s...", query[:30])

        facts = []
        edges_result = []
        nodes_result = []

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('\uff0c', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))

                scored_edges.sort(key=lambda x: x[0], reverse=True)

                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })

            if scope in ["nodes", "both"]:
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))

                scored_nodes.sort(key=lambda x: x[0], reverse=True)

                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info("Local search complete: found %d related facts", len(facts))

        except Exception as e:
            logger.error("Local search failed: %s", str(e))

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts),
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Get all graph nodes."""
        logger.info("Fetching all nodes for graph %s...", graph_id)

        driver = self.graphiti.driver
        records = run_async(
            driver.execute_query(
                """
                MATCH (n:Entity {group_id: $gid})
                RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                       n.summary AS summary
                """,
                {"gid": graph_id},
            )
        )

        result = []
        for record in records:
            result.append(NodeInfo(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                labels=record.get("labels", []),
                summary=record.get("summary", ""),
                attributes={},
            ))

        logger.info("Fetched %d nodes", len(result))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """Get all graph edges (including temporal information)."""
        logger.info("Fetching all edges for graph %s...", graph_id)

        driver = self.graphiti.driver
        records = run_async(
            driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO {group_id: $gid}]->(t:Entity)
                RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact,
                       s.uuid AS source_node_uuid, t.uuid AS target_node_uuid,
                       e.created_at AS created_at, e.valid_at AS valid_at,
                       e.invalid_at AS invalid_at, e.expired_at AS expired_at
                """,
                {"gid": graph_id},
            )
        )

        result = []
        for record in records:
            edge_info = EdgeInfo(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                fact=record.get("fact", ""),
                source_node_uuid=record.get("source_node_uuid", ""),
                target_node_uuid=record.get("target_node_uuid", ""),
            )

            if include_temporal:
                edge_info.created_at = str(record.get("created_at", "")) if record.get("created_at") else None
                edge_info.valid_at = str(record.get("valid_at", "")) if record.get("valid_at") else None
                edge_info.invalid_at = str(record.get("invalid_at", "")) if record.get("invalid_at") else None
                edge_info.expired_at = str(record.get("expired_at", "")) if record.get("expired_at") else None

            result.append(edge_info)

        logger.info("Fetched %d edges", len(result))
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """Get detailed information for a single node."""
        logger.info("Getting node details: %s...", node_uuid[:8])

        try:
            driver = self.graphiti.driver
            records = self._call_with_retry(
                func=lambda: run_async(
                    driver.execute_query(
                        """
                        MATCH (n:Entity {uuid: $uuid})
                        RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                               n.summary AS summary
                        """,
                        {"uuid": node_uuid},
                    )
                ),
                operation_name=f"Get node details (uuid={node_uuid[:8]}...)",
            )

            if not records:
                return None

            record = records[0]
            return NodeInfo(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                labels=record.get("labels", []),
                summary=record.get("summary", ""),
                attributes={},
            )
        except Exception as e:
            logger.error("Failed to get node details: %s", str(e))
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """Get all edges related to a node."""
        logger.info("Getting edges related to node %s...", node_uuid[:8])

        try:
            all_edges = self.get_all_edges(graph_id)
            result = []
            for edge in all_edges:
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)

            logger.info("Found %d edges related to the node", len(result))
            return result

        except Exception as e:
            logger.warning("Failed to get node edges: %s", str(e))
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """Get entities by type."""
        logger.info("Getting entities of type %s...", entity_type)

        all_nodes = self.get_all_nodes(graph_id)
        filtered = [node for node in all_nodes if entity_type in node.labels]

        logger.info("Found %d entities of type %s", len(filtered), entity_type)
        return filtered

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """Get the relationship summary for a specified entity."""
        logger.info("Getting relationship summary for entity %s...", entity_name)

        search_result = self.search_graph(graph_id=graph_id, query=entity_name, limit=20)

        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Get graph statistics."""
        logger.info("Getting statistics for graph %s...", graph_id)

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        skip_labels = {"Entity", "Node", "EntityNode", "EpisodicNode"}
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in skip_labels:
                    entity_types[label] = entity_types.get(label, 0) + 1

        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30,
    ) -> Dict[str, Any]:
        """Get simulation-related context information."""
        logger.info("Getting simulation context: %s...", simulation_requirement[:50])

        search_result = self.search_graph(graph_id=graph_id, query=simulation_requirement, limit=limit)
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)

        skip_labels = {"Entity", "Node", "EntityNode", "EpisodicNode"}
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in skip_labels]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary,
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    # ========== Core retrieval tools ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """InsightForge - Deep insight retrieval."""
        logger.info("InsightForge: %s...", query[:50])

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[],
        )

        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info("Generated %d sub-questions", len(sub_queries))

        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(graph_id=graph_id, query=sub_query, limit=15, scope="edges")
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            all_edges.extend(search_result.edges)

        main_search = self.search_graph(graph_id=graph_id, query=query, limit=20, scope="edges")
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        entity_insights = []
        node_map = {}

        for uuid_val in list(entity_uuids):
            if not uuid_val:
                continue
            try:
                node = self.get_node_detail(uuid_val)
                if node:
                    node_map[uuid_val] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node", "EntityNode", "EpisodicNode"]), "Entity")
                    related_facts = [f for f in all_facts if node.name.lower() in f.lower()]
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts,
                    })
            except Exception as e:
                logger.debug("Failed to fetch node %s: %s", uuid_val, e)
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(
            "InsightForge complete: %d facts, %d entities, %d relationships",
            result.total_facts, result.total_entities, result.total_relationships,
        )
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """Generate sub-questions with an LLM."""
        system_prompt = """You are a professional question analysis expert. Your task is to decompose a complex question into multiple sub-questions that can be independently observed in a simulated world.

Requirements:
1. Each sub-question should be specific enough to find related Agent behaviors or events in the simulation.
2. Sub-questions should cover different dimensions of the original question (who, what, why, how, when, where).
3. Sub-questions should be relevant to the simulation scenario.
4. Return JSON format: {"sub_queries": ["Sub-question 1", "Sub-question 2", ...]}"""

        user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please decompose the following question into {max_queries} sub-questions:
{query}

Return a JSON list of sub-questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning("Failed to generate sub-questions: %s; using defaults", str(e))
            return [
                query,
                f"Major participants in {query}",
                f"Causes and impacts of {query}",
                f"How {query} evolved",
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
    ) -> PanoramaResult:
        """PanoramaSearch - Panoramic search."""
        logger.info("PanoramaSearch: %s...", query[:50])

        result = PanoramaResult(query=query)

        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]

            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('\uff0c', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info("PanoramaSearch complete: %d active, %d historical", result.active_count, result.historical_count)
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """QuickSearch - Simple search."""
        logger.info("QuickSearch: %s...", query[:50])
        result = self.search_graph(graph_id=graph_id, query=query, limit=limit, scope="edges")
        logger.info("QuickSearch complete: %d results", result.total_count)
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """InterviewAgents - In-depth interview. Calls real OASIS interview API."""
        from .simulation_runner import SimulationRunner
        import re

        logger.info("InterviewAgents: %s...", interview_requirement[:50])

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning("No profile files found for simulation %s", simulation_id)
            result.summary = "No interviewable Agent profile files found"
            return result

        result.total_agents = len(profiles)
        logger.info("Loaded %d Agent profiles", len(profiles))

        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents,
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info("Selected %d Agents for interview: %s", len(selected_agents), selected_indices)

        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )
            logger.info("Generated %d interview questions", len(result.interview_questions))

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Please combine your persona with all your past memories and actions, "
            "and answer the following questions directly in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language; do not call any tools.\n"
            "2. Do not return JSON format or tool-call format.\n"
            "3. Do not use Markdown headings (such as #, ##, ###).\n"
            "4. Answer each question by number, and start each answer with 'Question X:' (X is the question number).\n"
            "5. Separate answers for each question with blank lines.\n"
            "6. Provide substantive content, at least 2-3 sentences per question.\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt,
                })

            logger.info("Calling batch interview API: %d Agents", len(interviews_request))

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0,
            )

            logger.info("Interview API returned: %d results", api_result.get('interviews_count', 0))

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning("Interview API returned failure: %s", error_msg)
                result.summary = f"Interview API call failed: {error_msg}"
                return result

            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = self._clean_tool_call_response(twitter_result.get("response", ""))
                reddit_response = self._clean_tool_call_response(reddit_result.get("response", ""))

                twitter_text = twitter_response if twitter_response else "(No response from this platform)"
                reddit_text = reddit_response if reddit_response else "(No response from this platform)"
                response_text = f"[Twitter response]\n{twitter_text}\n\n[Reddit response]\n{reddit_text}"

                combined_responses = f"{twitter_response} {reddit_response}"
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Question\s*\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "." for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            logger.warning("Interview API call failed: %s", e)
            result.summary = f"Interview failed: {str(e)}"
            return result
        except Exception as e:
            logger.error("Interview API call exception: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"An error occurred during the interview process: {str(e)}"
            return result

        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info("InterviewAgents complete: interviewed %d Agents", result.interviewed_count)
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Clean JSON tool-call wrappers in Agent replies."""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load Agent profile files for a simulation."""
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}',
        )

        profiles = []

        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info("Loaded %d profiles from reddit_profiles.json", len(profiles))
                return profiles
            except Exception as e:
                logger.warning("Failed to read reddit_profiles.json: %s", e)

        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown",
                        })
                logger.info("Loaded %d profiles from twitter_profiles.csv", len(profiles))
                return profiles
            except Exception as e:
                logger.warning("Failed to read twitter_profiles.csv: %s", e)

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        """Use an LLM to select Agents for interview."""
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", []),
            }
            agent_summaries.append(summary)

        system_prompt = """You are a professional interview planning expert. Select the most suitable interviewees from a list of simulated Agents.

Selection criteria:
1. The Agent's identity/profession is relevant to the interview topic
2. The Agent may hold unique or valuable viewpoints
3. Select diverse perspectives
4. Prioritize roles directly related to the event

Return JSON format:
{
    "selected_indices": [list of selected Agent indices],
    "reasoning": "Selection reasoning"
}"""

        user_prompt = f"""Interview requirement:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

Available Agents (total {len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} most suitable Agents."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Auto-selected based on relevance")

            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning("LLM failed to select Agents; using default selection: %s", e)
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate interview questions with an LLM."""
        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]

        system_prompt = """You are a professional journalist/interviewer. Generate 3-5 in-depth interview questions.

Question requirements:
1. Ask open-ended questions that encourage detailed responses
2. Questions should allow different answers across roles
3. Cover multiple dimensions such as facts, opinions, and feelings
4. Use natural language like a real interview
5. Keep each question concise and clear

Return JSON format: {"questions": ["Question 1", "Question 2", ...]}"""

        user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee roles: {', '.join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )
            return response.get("questions", [f"What are your views on {interview_requirement}?"])

        except Exception as e:
            logger.warning("Failed to generate interview questions: %s", e)
            return [
                f"What is your perspective on {interview_requirement}?",
                "How does this matter affect you or the group you represent?",
                "How do you think this issue should be resolved or improved?",
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str,
    ) -> str:
        """Generate interview summary."""
        if not interviews:
            return "No interviews were completed"

        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}")

        system_prompt = """You are a professional news editor. Generate an interview summary based on responses from multiple interviewees.

Summary requirements:
1. Distill the main viewpoints from each side
2. Point out consensus and disagreements
3. Highlight valuable quotes
4. Stay objective and neutral
5. Keep it concise

Format constraints:
- Use plain-text paragraphs
- Do not use Markdown headings
- Do not use separators"""

        user_prompt = f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate an interview summary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return summary

        except Exception as e:
            logger.warning("Failed to generate interview summary: %s", e)
            return f"Interviewed {len(interviews)} interviewees, including: " + ", ".join([i.agent_name for i in interviews])


# Backwards-compatible alias
ZepToolsService = GraphToolsService
