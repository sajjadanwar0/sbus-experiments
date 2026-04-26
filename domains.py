DOMAINS = [
    {
        "name": "ecommerce",
        "description": (
            "Design a data pipeline for a high-throughput e-commerce platform "
            "processing order events, inventory updates, and customer activity. "
            "Order events are write-heavy and require transactional guarantees: "
            "an order must not be lost or duplicated. Inventory updates are "
            "moderate-volume but require strong consistency for stock-tracking. "
            "Customer activity (clicks, views, cart abandonment) is high-volume "
            "but can tolerate eventual consistency. Peak load: 50K events/second."
        ),
        "pressures": ["transactional_orders", "moderate_consistency_inventory",
                      "eventual_consistency_activity", "high_throughput"],
    },
    {
        "name": "healthcare",
        "description": (
            "Design a data pipeline for a hospital information system processing "
            "patient records, lab results, medication administration events, and "
            "clinical trial data. Compliance requirements (HIPAA, GDPR for EU "
            "patients) require full audit trails on every access and "
            "modification. Lab results must be queryable within 60 seconds of "
            "production. Patient records must be replicated across geographies "
            "for disaster recovery. Peak load: 500 events/second."
        ),
        "pressures": ["audit_compliance", "low_latency_queries",
                      "geo_replication", "encryption_at_rest"],
    },
    {
        "name": "iot",
        "description": (
            "Design a data pipeline for an IoT platform ingesting sensor data "
            "from 1M+ connected devices: industrial machinery, smart meters, "
            "vehicle telemetry. Devices send metrics every 1-60 seconds with "
            "highly variable payload sizes. Time-series storage must support "
            "sub-second query latency for dashboards and minute-level "
            "aggregation for analytics. Devices may go offline for extended "
            "periods and need backfill on reconnect. Peak load: 100K events/sec."
        ),
        "pressures": ["high_ingest_rate", "time_series_optimised",
                      "backfill_on_reconnect", "downsampling_required"],
    },
    {
        "name": "financial",
        "description": (
            "Design a data pipeline for a high-frequency trading platform "
            "processing market quotes, order events, and trade executions. "
            "Latency requirements are sub-millisecond end-to-end. Every event "
            "must be processed exactly once for regulatory compliance and "
            "P&L correctness. Settlement reconciliation requires querying "
            "historical events with full lineage. Peak load: 1M events/second "
            "during market open."
        ),
        "pressures": ["sub_ms_latency", "exactly_once_semantics",
                      "regulatory_lineage", "settlement_reconciliation"],
    },
    {
        "name": "social",
        "description": (
            "Design a data pipeline for a social media platform processing "
            "posts, likes, follows, and comments at massive scale. Eventual "
            "consistency is acceptable for engagement counters (likes, "
            "comments) but social graph reads (who-follows-whom) require "
            "strong consistency for personalisation. Hot-spot detection: a "
            "single celebrity post may receive 100K likes/minute. Peak load: "
            "500K events/second globally."
        ),
        "pressures": ["massive_scale", "eventual_engagement_counters",
                      "strong_consistency_graph", "hotspot_handling"],
    },
    {
        "name": "gaming",
        "description": (
            "Design a data pipeline for an online multiplayer gaming platform "
            "processing player telemetry, match events, achievements, and "
            "in-game purchases. Telemetry is high-cardinality (per-player, "
            "per-match, per-event-type). Match events must be processable in "
            "real-time for live leaderboards. Achievements require historical "
            "queries across player history. Purchases need transactional "
            "guarantees. Peak load: 200K events/second during evening peak."
        ),
        "pressures": ["high_cardinality", "real_time_leaderboards",
                      "historical_queries", "transactional_purchases"],
    },
    {
        "name": "supply_chain",
        "description": (
            "Design a data pipeline for a B2B supply chain platform "
            "ingesting EDI documents, RFID scans, GPS tracking, warehouse "
            "events, and customs filings from heterogeneous sources. Sources "
            "include SOAP APIs, SFTP drops, partner-specific REST endpoints, "
            "and legacy mainframe feeds. Schema variation is high — each "
            "partner has a different format. Peak load: variable, with "
            "weekly bursts up to 10K events/second."
        ),
        "pressures": ["heterogeneous_sources", "schema_variation",
                      "burst_handling", "b2b_partner_apis"],
    },
    {
        "name": "ad_tech",
        "description": (
            "Design a data pipeline for an ad-tech platform processing bid "
            "requests, impressions, clicks, and conversions. Bid requests "
            "must complete in <50ms end-to-end (RTB latency budget). User "
            "profile lookups during bidding require sub-millisecond reads. "
            "Conversion attribution requires joining click and impression "
            "events over windows up to 30 days. Peak load: 2M bid "
            "requests/second."
        ),
        "pressures": ["sub_50ms_rtb_latency", "low_latency_user_lookups",
                      "30_day_attribution_window", "high_qps"],
    },
]

INITIAL_SHARDS = {
    "ingestion": (
        "INGESTION COMPONENT (TBD)\n"
        "Responsibility: how events enter the pipeline, including\n"
        "transport protocol, buffering strategy, and back-pressure\n"
        "handling. Owner agent will specify concrete choices.\n"
    ),
    "transformation": (
        "TRANSFORMATION COMPONENT (TBD)\n"
        "Responsibility: how raw events are processed, including\n"
        "stream-vs-batch model, schema validation, enrichment, and\n"
        "windowing. Owner agent will specify concrete choices.\n"
    ),
    "storage": (
        "STORAGE COMPONENT (TBD)\n"
        "Responsibility: where processed events are persisted, including\n"
        "data store technology, partitioning scheme, retention policy,\n"
        "and access patterns. Owner agent will specify concrete choices.\n"
    ),
    "monitoring": (
        "MONITORING COMPONENT (TBD)\n"
        "Responsibility: how the pipeline is observed, including\n"
        "metrics collection, alerting thresholds, log aggregation, and\n"
        "tracing strategy. Owner agent will specify concrete choices.\n"
    ),
}

SHARD_KEYS = list(INITIAL_SHARDS.keys())

AGENTS_PER_TRIAL = SHARD_KEYS
