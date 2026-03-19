"""
Entity read and filter service.

Reads nodes from the Graphiti/Neo4j graph and filters nodes that match
predefined entity types. Keeps the same public interface as the original
Zep-based implementation.
"""

import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from ..config import Config
from ..utils.logger import get_logger
from ..utils.graphiti_client import get_graphiti_sync, run_async

logger = get_logger('mirofish.entity_reader')

T = TypeVar('T')


@dataclass
class EntityNode:
    """Entity node data structure"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Get entity type (excluding default Entity label)"""
        for label in self.labels:
            if label not in ["Entity", "Node", "EntityNode", "EpisodicNode"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Filtered entity collection"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class EntityReader:
    """
    Entity read and filter service.

    Main features:
    1. Read all nodes from the Graphiti/Neo4j graph
    2. Filter nodes that match predefined entity types
    3. Get related edges and associated node information for each entity
    """

    def __init__(self):
        self.graphiti = get_graphiti_sync()

    def _call_with_retry(
        self,
        func: Callable[[], T],
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
    ) -> T:
        """Execute with retry and exponential backoff."""
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "%s attempt %d failed: %s, retrying in %.1fs...",
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

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all entity nodes of the graph."""
        logger.info("Fetching all nodes for graph %s...", graph_id)

        driver = self.graphiti.driver
        records = run_async(
            driver.execute_query(
                """
                MATCH (n:Entity {group_id: $gid})
                RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                       n.summary AS summary, n.created_at AS created_at
                """,
                {"gid": graph_id},
            )
        )

        nodes_data = []
        for record in records:
            nodes_data.append({
                "uuid": record.get("uuid", ""),
                "name": record.get("name", ""),
                "labels": record.get("labels", []),
                "summary": record.get("summary", ""),
                "attributes": {},
            })

        logger.info("Fetched %d nodes in total", len(nodes_data))
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all edges of the graph."""
        logger.info("Fetching all edges for graph %s...", graph_id)

        driver = self.graphiti.driver
        records = run_async(
            driver.execute_query(
                """
                MATCH (s:Entity)-[e:RELATES_TO {group_id: $gid}]->(t:Entity)
                RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact,
                       s.uuid AS source_node_uuid, t.uuid AS target_node_uuid
                """,
                {"gid": graph_id},
            )
        )

        edges_data = []
        for record in records:
            edges_data.append({
                "uuid": record.get("uuid", ""),
                "name": record.get("name", ""),
                "fact": record.get("fact", ""),
                "source_node_uuid": record.get("source_node_uuid", ""),
                "target_node_uuid": record.get("target_node_uuid", ""),
                "attributes": {},
            })

        logger.info("Fetched %d edges in total", len(edges_data))
        return edges_data

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Get all related edges for the specified node."""
        try:
            driver = self.graphiti.driver
            records = self._call_with_retry(
                func=lambda: run_async(
                    driver.execute_query(
                        """
                        MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(other:Entity)
                        RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact,
                               startNode(e).uuid AS source_node_uuid,
                               endNode(e).uuid AS target_node_uuid
                        """,
                        {"uuid": node_uuid},
                    )
                ),
                operation_name=f"get node edges(node={node_uuid[:8]}...)",
            )

            edges_data = []
            for record in records:
                edges_data.append({
                    "uuid": record.get("uuid", ""),
                    "name": record.get("name", ""),
                    "fact": record.get("fact", ""),
                    "source_node_uuid": record.get("source_node_uuid", ""),
                    "target_node_uuid": record.get("target_node_uuid", ""),
                    "attributes": {},
                })

            return edges_data
        except Exception as e:
            logger.warning("Failed to get edges for node %s: %s", node_uuid, str(e))
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """
        Filter nodes that match predefined entity types.

        Same logic as before: nodes with labels beyond "Entity"/"Node"
        are considered to match predefined types.
        """
        logger.info("Starting to filter entities for graph %s...", graph_id)

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities = []
        entity_types_found = set()

        skip_labels = {"Entity", "Node", "EntityNode", "EpisodicNode"}

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in skip_labels]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(
            "Filter complete: total nodes %d, matching %d, entity types: %s",
            total_count, len(filtered_entities), entity_types_found,
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str,
    ) -> Optional[EntityNode]:
        """Get a single entity with full context (edges and associated nodes)."""
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
                        {"uuid": entity_uuid},
                    )
                ),
                operation_name=f"get node details(uuid={entity_uuid[:8]}...)",
            )

            if not records:
                return None

            node = records[0]

            edges = self.get_node_edges(entity_uuid)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}

            related_edges = []
            related_node_uuids = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })

            return EntityNode(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes={},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error("Failed to get entity %s: %s", entity_uuid, str(e))
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """Get all entities of the specified type."""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities


# Backwards-compatible alias so existing imports still work
ZepEntityReader = EntityReader
