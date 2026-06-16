"""External-integration contract layer for SOTA-Paddle-MTMC.

This package holds the compatibility shim that makes the new
pipeline emit the same MQTT topics, payload fields, MinIO bucket
and prefix, stream paths, and ROI zones as the legacy
``Service/offline-people-counting`` pipeline.

Nothing in this package should be required for the core detection
/ ReID / resolver logic to work — it is a pure adapter on top of
the existing telemetry / storage / streaming surfaces.
"""
