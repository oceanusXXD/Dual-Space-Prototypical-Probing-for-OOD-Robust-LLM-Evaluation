# Hidden State

The hidden-state layer is the first algorithm stage. It owns model extraction,
cache metadata, layer selection, pooling, and view definitions for A-space and
B-space representations. Downstream classifiers and detectors consume its
output contract rather than calling model-specific extraction code directly.
