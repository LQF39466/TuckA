# Implementation Walkthrough

### Code Structure
A Hugging Face PEFT adapter is composed of three modules: `config.py` for setting up parameters, `model.py` for adapter injection, and `layer.py` for the linear layer itself.

### Built-in Comparisons & Logging
The code in `layer.py` is mostly consistent with the algorithm pseudocode in our paper (refer to Appendix B). We include extra configurations that allow you to run comparisons between different initialization methods. There is also logging-related code that dumps the data used for visualizations in our paper.

### Routing (Important)
Batch-level routing is a key component of TuckA. The router assignment is implemented in `model.py` during the adapter injection process. Inspired by the implementation of weight sharing in VeRA, the routing result is passed to subsequent modules via `BufferDict`. It is worth noting that this sequential operation requires the router to be the **first adapted module** in the forward pass. This is usually guaranteed, as the enumeration of modules often follows the forward pass order. However, this is not the case for certain models (e.g., CLIP), so you must manually assign the router module's name by passing `router_name` to the `TuckAConfig` class for them.