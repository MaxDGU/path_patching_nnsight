#!/usr/bin/env python3
"""
Path Patching with NNsight - A Tutorial Implementation
=======================================================

This module teaches you how to do activation patching and path patching
using NNsight on GPT-2 style models. It's designed to be readable and
educational, not just functional.

Key Concepts:
- Node: A location in the model where we can read/write activations
- Activation Patching: Replace one activation with another to measure importance
- Path Patching: Measure the effect of a specific pathway through the model

NNsight Basics:
- NNsight wraps HuggingFace models and lets you intervene on activations
- Use `with model.trace() as tracer:` to set up an intervention
- Use `with tracer.invoke(input):` to run the model with interventions
- Access activations like `model.transformer.h[0].attn.c_proj.output`
- Use `.save()` to capture a value, `.value` to read it after the trace

Example:
    >>> model = LanguageModel("gpt2", device_map="cpu")
    >>> with model.trace() as tracer:
    ...     with tracer.invoke("Hello world"):
    ...         hidden = model.transformer.h[5].output[0].save()
    >>> print(hidden.value.shape)  # torch.Size([1, 2, 768])
"""

import torch as t
from torch import Tensor
from typing import Optional, Union, Dict, Callable, List, Tuple, Any
from typing_extensions import Literal
from dataclasses import dataclass

# Optional: jaxtyping for nice type hints, but not required
try:
    from jaxtyping import Float, Int
except ImportError:
    Float = Int = lambda x, y: Tensor

from nnsight import LanguageModel


# =============================================================================
# Helper Functions
# =============================================================================

def get_value(obj):
    """
    Extract the actual tensor from an nnsight proxy object.

    After a trace completes, saved values are wrapped in proxy objects.
    This helper handles both proxies (with .value) and raw tensors.
    """
    return obj.value if hasattr(obj, 'value') else obj


# Type aliases for cleaner signatures
SeqPos = Optional[Union[int, List[int], Tensor]]
IterSeqPos = Union[SeqPos, Literal["each"]]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GPT2Config:
    """
    Configuration for GPT-2 style models.

    These are the architectural parameters we need to know for patching.
    For GPT-2 small, the defaults are correct. For other models, you'll
    need to adjust these values.
    """
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_head: int = 64
    d_mlp: int = 3072


# =============================================================================
# Node: Specifying Where to Patch
# =============================================================================

class Node:
    """
    A Node represents a specific activation location in the model.

    Think of the transformer as a graph. Each Node is a point where we can
    tap into the computation - reading what's there or replacing it with
    something else.

    Available locations in GPT-2 (via nnsight):

        attn_out   - Output of attention block (after the output projection)
        mlp_out    - Output of MLP block (after the output projection)
        resid_post - The residual stream after the whole layer
        ln1_out    - Output of first LayerNorm (input to attention)
        ln2_out    - Output of second LayerNorm (input to MLP)

    Examples:
        Node("attn_out", layer=0)      # Attention output at layer 0
        Node("mlp_out", layer=5)       # MLP output at layer 5
        Node("resid_post", layer=11)   # Final residual stream

    Why these specific locations?
        NNsight accesses HuggingFace model internals directly. GPT-2 uses
        combined QKV projection (c_attn), so we can't easily access individual
        Q, K, V, or per-head outputs. We're limited to module outputs.
    """

    # These are the only locations we can reliably access in nnsight
    VALID_NODES = ["attn_out", "mlp_out", "resid_post", "ln1_out", "ln2_out"]

    def __init__(
        self,
        component_name: str,
        layer: Optional[int] = None,
        seq_pos: SeqPos = None,
    ):
        self.component_name = component_name
        self.layer = layer
        self.seq_pos = seq_pos

        # Validate inputs
        if layer is not None:
            assert layer >= 0, f"Layer must be non-negative, got {layer}"

        assert component_name in self.VALID_NODES, \
            f"Unknown component '{component_name}'. Choose from: {self.VALID_NODES}"

    def __repr__(self):
        parts = [f"Node('{self.component_name}'"]
        if self.layer is not None:
            parts.append(f"layer={self.layer}")
        if self.seq_pos is not None:
            parts.append(f"seq_pos={self.seq_pos}")
        return ", ".join(parts) + ")"

    def get_position(self) -> float:
        """
        Get this node's position in the forward pass.

        Used to check causal ordering - you can only patch from earlier
        nodes to later nodes. Returns a float where higher = later in
        the computation.
        """
        layer = self.layer if self.layer is not None else 0

        # Order within a layer: ln1 -> attn -> ln2 -> mlp -> resid_post
        component_order = {
            "ln1_out": 0.1,
            "attn_out": 0.3,
            "ln2_out": 0.4,
            "mlp_out": 0.6,
            "resid_post": 1.0,
        }

        return layer + component_order.get(self.component_name, 0.5)

    def __lt__(self, other: "Node") -> bool:
        return self.get_position() < other.get_position()

    def __le__(self, other: "Node") -> bool:
        return self.get_position() <= other.get_position()

    def get_activation(self, model: LanguageModel, layer_module) -> Any:
        """
        Get the activation proxy for this node during a trace.

        This is where the nnsight magic happens. We reach into the model's
        internals and grab the activation at our specified location.

        Args:
            model: The nnsight LanguageModel (not used directly, but kept for API)
            layer_module: The transformer block, e.g., model.transformer.h[5]

        Returns:
            An nnsight proxy that you can .save() or modify
        """
        if self.component_name == "attn_out":
            # Attention output comes from c_proj (the output projection)
            return layer_module.attn.c_proj.output

        elif self.component_name == "mlp_out":
            # MLP output comes from c_proj (the final projection)
            return layer_module.mlp.c_proj.output

        elif self.component_name == "ln1_out":
            # First layer norm, fed into attention
            return layer_module.ln_1.output

        elif self.component_name == "ln2_out":
            # Second layer norm, fed into MLP
            return layer_module.ln_2.output

        elif self.component_name == "resid_post":
            # Block output is a tuple, first element is hidden states
            return layer_module.output[0]

        else:
            raise ValueError(f"Unknown component: {self.component_name}")


# =============================================================================
# IterNode: Iterating Over Multiple Nodes
# =============================================================================

class IterNode:
    """
    An iterator that generates multiple Nodes for batch experiments.

    When you want to test patching at every layer, or every component,
    use IterNode instead of writing a loop yourself.

    Examples:
        IterNode("attn_out")              # All 12 layers of attn_out
        IterNode(["attn_out", "mlp_out"]) # Both components, all layers
    """

    def __init__(
        self,
        node_names: Union[str, List[str]],
        seq_pos: IterSeqPos = None,
    ):
        self.node_names = [node_names] if isinstance(node_names, str) else node_names
        self.seq_pos = seq_pos

    def iterate(self, n_layers: int, n_heads: int = 12, seq_len: Optional[int] = None):
        """
        Generate all (key, Node) pairs for this iterator.

        Yields tuples of (identifier, Node) where identifier can be used
        to organize results.
        """
        for node_name in self.node_names:
            for layer in range(n_layers):
                if self.seq_pos == "each" and seq_len is not None:
                    for pos in range(seq_len):
                        yield (node_name, layer, pos), Node(node_name, layer, seq_pos=pos)
                else:
                    yield (node_name, layer), Node(node_name, layer, seq_pos=self.seq_pos)


# =============================================================================
# Caching Activations
# =============================================================================

def cache_activations(
    model: LanguageModel,
    input_data: Union[str, List[str], Dict[str, Tensor]],
    nodes: List[Node],
) -> Dict[Tuple, Tensor]:
    """
    Run the model and save activations at specified nodes.

    This is often the first step in patching: run on your "source" input
    and cache the activations you want to patch in later.

    Args:
        model: NNsight LanguageModel
        input_data: Prompts (str or list) or tokenized input (dict)
        nodes: List of Nodes specifying which activations to cache

    Returns:
        Dict mapping (component_name, layer) -> activation tensor

    Example:
        >>> nodes = [Node("attn_out", layer=5), Node("mlp_out", layer=5)]
        >>> cache = cache_activations(model, ["Hello world"], nodes)
        >>> cache[("attn_out", 5)].shape
        torch.Size([1, 2, 768])

    Important: NNsight requires accessing activations in forward-pass order.
    We sort nodes automatically, but if you're doing manual traces, access
    layer 0 before layer 1, and attn before mlp within a layer.
    """
    cache = {}

    # NNsight quirk: must access activations in the order they're computed
    # Otherwise you get "OutOfOrderError: Value was missed"
    sorted_nodes = sorted(nodes, key=lambda n: n.get_position())

    with model.trace() as tracer:
        with tracer.invoke(input_data):
            for node in sorted_nodes:
                layer_module = model.transformer.h[node.layer]
                activation = node.get_activation(model, layer_module).save()
                cache[(node.component_name, node.layer)] = activation

    # Convert proxies to actual tensors
    return {k: get_value(v) for k, v in cache.items()}


# =============================================================================
# Activation Patching
# =============================================================================

def act_patch(
    model: LanguageModel,
    orig_input: Union[str, List[str], Dict[str, Tensor]],
    new_cache: Dict[Tuple, Tensor],
    patching_nodes: Union[Node, IterNode, List[Node]],
    patching_metric: Callable[[Tensor], Tensor],
    seq_pos: SeqPos = None,
    config: Optional[GPT2Config] = None,
    verbose: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    """
    Activation patching: measure a component's importance by replacing it.

    The idea: run on orig_input, but at patching_nodes, swap in activations
    from new_cache. Then measure how much the output changes.

    Common setup (noising):
        - orig_input: clean prompts (model gets right answer)
        - new_cache: from corrupted prompts (model gets wrong answer)
        - Large change in metric = this component was important

    Args:
        model: NNsight LanguageModel
        orig_input: The input to run the model on
        new_cache: Cached activations to patch in (from cache_activations)
        patching_nodes: Where to patch (Node, IterNode, or list of Nodes)
        patching_metric: Function: logits -> scalar (e.g., logit difference)
        seq_pos: Optional position(s) to patch at
        config: Model config (auto-detected if None)
        verbose: Show progress bar

    Returns:
        If patching_nodes is a single Node: scalar Tensor
        If patching_nodes is IterNode: Dict[component_name, Tensor of results]

    Example:
        >>> cache = cache_activations(model, corrupt_prompts, [Node("attn_out", 9)])
        >>> result = act_patch(
        ...     model, clean_prompts, cache,
        ...     Node("attn_out", layer=9),
        ...     lambda logits: logits[:, -1, target_token].mean()
        ... )
    """
    if config is None:
        config = GPT2Config(
            n_layers=model.config.n_layer,
            n_heads=model.config.n_head,
            d_model=model.config.n_embd,
        )

    # Single node -> return scalar
    if isinstance(patching_nodes, Node):
        return _do_patch(model, orig_input, new_cache, patching_nodes, patching_metric, seq_pos)

    # IterNode -> iterate and collect results
    elif isinstance(patching_nodes, IterNode):
        results = {}
        iterator = list(patching_nodes.iterate(config.n_layers, config.n_heads))

        if verbose:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Activation patching")

        for key, node in iterator:
            results[key] = _do_patch(model, orig_input, new_cache, node, patching_metric, seq_pos)

        return _reshape_results(results, patching_nodes, config)

    # List of nodes -> patch all at once
    elif isinstance(patching_nodes, list):
        return _do_patch(model, orig_input, new_cache, patching_nodes, patching_metric, seq_pos)

    else:
        raise TypeError(f"patching_nodes must be Node, IterNode, or list, got {type(patching_nodes)}")


def _do_patch(
    model: LanguageModel,
    orig_input: Union[str, List[str], Dict[str, Tensor]],
    new_cache: Dict[Tuple, Tensor],
    nodes: Union[Node, List[Node]],
    patching_metric: Callable[[Tensor], Tensor],
    seq_pos: SeqPos = None,
) -> Tensor:
    """Actually perform the patch and return the metric."""
    if isinstance(nodes, Node):
        nodes = [nodes]

    with model.trace() as tracer:
        with tracer.invoke(orig_input):
            for node in nodes:
                layer = model.transformer.h[node.layer]
                key = (node.component_name, node.layer)

                if key not in new_cache:
                    raise KeyError(f"Node {node} not in cache. Have: {list(new_cache.keys())}")

                patch_value = new_cache[key]

                # Patch the activation - this is where the intervention happens!
                # We assign the cached value to overwrite what the model computed
                activation = node.get_activation(model, layer)

                if seq_pos is not None:
                    activation[:, seq_pos, :] = patch_value[:, seq_pos, :]
                else:
                    activation[:] = patch_value

            logits = model.lm_head.output.save()

    return patching_metric(get_value(logits))


# =============================================================================
# Path Patching
# =============================================================================

def path_patch(
    model: LanguageModel,
    orig_input: Union[str, List[str], Dict[str, Tensor]],
    new_input: Union[str, List[str], Dict[str, Tensor]],
    sender_nodes: Union[Node, IterNode, List[Node]],
    receiver_nodes: Union[Node, List[Node]],
    patching_metric: Callable[[Tensor], Tensor],
    seq_pos: SeqPos = None,
    config: Optional[GPT2Config] = None,
    verbose: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    """
    Path patching: measure the effect of a specific pathway.

    Unlike activation patching (which measures total importance), path
    patching asks: "How much does information flow from sender to receiver?"

    Algorithm:
        1. Cache activations on orig_input and new_input
        2. Run on orig_input, but at sender_nodes, use new_input's activations
        3. Measure the effect on the output

    This tells you whether the sender is sending important information
    that affects the final output.

    Args:
        model: NNsight LanguageModel
        orig_input: Original (typically clean) input
        new_input: Alternative (typically corrupted) input
        sender_nodes: Where the path starts
        receiver_nodes: Where the path ends (must be later than sender)
        patching_metric: Function: logits -> scalar
        seq_pos: Optional position(s) to patch
        config: Model config (auto-detected if None)
        verbose: Show progress bar

    Returns:
        Patching results (scalar or dict of tensors)

    Example:
        >>> result = path_patch(
        ...     model,
        ...     clean_prompts,
        ...     corrupt_prompts,
        ...     sender_nodes=Node("attn_out", layer=5),
        ...     receiver_nodes=Node("resid_post", layer=11),
        ...     patching_metric=logit_diff_metric,
        ... )
    """
    if config is None:
        config = GPT2Config(
            n_layers=model.config.n_layer,
            n_heads=model.config.n_head,
            d_model=model.config.n_embd,
        )

    # Cache activations from both inputs
    if verbose:
        print("Caching original activations...")

    all_sender_nodes = _collect_nodes(sender_nodes, config)
    orig_cache = cache_activations(model, orig_input, all_sender_nodes)
    new_cache = cache_activations(model, new_input, all_sender_nodes)

    # Route based on sender type
    if isinstance(sender_nodes, Node):
        return _do_path_patch(
            model, orig_input, sender_nodes, receiver_nodes,
            patching_metric, orig_cache, new_cache, seq_pos
        )

    elif isinstance(sender_nodes, IterNode):
        results = {}
        iterator = list(sender_nodes.iterate(config.n_layers, config.n_heads))

        if verbose:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Path patching")

        for key, node in iterator:
            results[key] = _do_path_patch(
                model, orig_input, node, receiver_nodes,
                patching_metric, orig_cache, new_cache, seq_pos
            )

        return _reshape_results(results, sender_nodes, config)

    elif isinstance(sender_nodes, list):
        return _do_path_patch(
            model, orig_input, sender_nodes, receiver_nodes,
            patching_metric, orig_cache, new_cache, seq_pos
        )

    else:
        raise TypeError(f"sender_nodes must be Node, IterNode, or list")


def _do_path_patch(
    model: LanguageModel,
    orig_input: Union[str, List[str], Dict[str, Tensor]],
    sender: Union[Node, List[Node]],
    receiver: Union[Node, List[Node]],
    patching_metric: Callable[[Tensor], Tensor],
    orig_cache: Dict[Tuple, Tensor],
    new_cache: Dict[Tuple, Tensor],
    seq_pos: SeqPos = None,
) -> Tensor:
    """Perform path patching for a single sender-receiver pair."""
    senders = [sender] if isinstance(sender, Node) else sender
    receivers = [receiver] if isinstance(receiver, Node) else receiver

    # Validate causal ordering - sender must come before receiver
    for s in senders:
        for r in receivers:
            if not (s < r):
                raise ValueError(
                    f"Invalid path: sender {s} must come before receiver {r}. "
                    f"Information can only flow forward in the model."
                )

    with model.trace() as tracer:
        with tracer.invoke(orig_input):
            # At sender locations, swap in the new_input activations
            for node in senders:
                layer = model.transformer.h[node.layer]
                key = (node.component_name, node.layer)
                patch_value = new_cache[key]

                activation = node.get_activation(model, layer)
                if seq_pos is not None:
                    activation[:, seq_pos, :] = patch_value[:, seq_pos, :]
                else:
                    activation[:] = patch_value

            logits = model.lm_head.output.save()

    return patching_metric(get_value(logits))


# =============================================================================
# Helper Functions
# =============================================================================

def _collect_nodes(nodes: Union[Node, IterNode, List[Node]], config: GPT2Config) -> List[Node]:
    """Expand any node specification into a flat list of Nodes."""
    if isinstance(nodes, Node):
        return [nodes]
    elif isinstance(nodes, IterNode):
        return [node for _, node in nodes.iterate(config.n_layers, config.n_heads)]
    elif isinstance(nodes, list):
        return nodes
    else:
        raise TypeError(f"Expected Node, IterNode, or list, got {type(nodes)}")


def _reshape_results(
    results: Dict[Tuple, Tensor],
    iter_node: IterNode,
    config: GPT2Config,
) -> Dict[str, Tensor]:
    """Reshape iteration results into nicely-shaped tensors."""
    output = {}

    for node_name in iter_node.node_names:
        # For layer-only iteration, shape is (n_layers,)
        tensor = t.zeros(config.n_layers)
        for layer in range(config.n_layers):
            key = (node_name, layer)
            if key in results:
                tensor[layer] = results[key]
        output[node_name] = tensor

    return output


# =============================================================================
# Metric Helpers
# =============================================================================

def logit_diff_metric(logits: Tensor, correct_idx: int, incorrect_idx: int, pos: int = -1) -> Tensor:
    """
    Simple logit difference metric.

    Returns: mean(logit[correct] - logit[incorrect]) at position `pos`.
    """
    return (logits[:, pos, correct_idx] - logits[:, pos, incorrect_idx]).mean()


def make_ioi_metric(correct_tokens: Tensor, incorrect_tokens: Tensor) -> Callable:
    """
    Create a metric for IOI-style tasks.

    Args:
        correct_tokens: Token IDs for correct answers, shape (batch,)
        incorrect_tokens: Token IDs for incorrect answers, shape (batch,)

    Returns:
        A function that computes mean(logit[correct] - logit[incorrect])
    """
    def metric(logits: Tensor) -> Tensor:
        final_logits = logits[:, -1, :]
        correct = final_logits.gather(dim=-1, index=correct_tokens.unsqueeze(-1)).squeeze(-1)
        incorrect = final_logits.gather(dim=-1, index=incorrect_tokens.unsqueeze(-1)).squeeze(-1)
        return (correct - incorrect).mean()
    return metric


# =============================================================================
# Tests (run with: python path_patching_nnsight.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    def run_tests():
        print("=" * 60)
        print("Path Patching Module - Test Suite")
        print("=" * 60)

        passed = failed = 0

        # Test 1: Node basics
        print("\n[1] Node class...")
        try:
            n1 = Node("attn_out", layer=0)
            n2 = Node("mlp_out", layer=5)
            n3 = Node("resid_post", layer=11)
            assert n1 < n2 < n3
            print(f"    Created: {n1}, {n2}, {n3}")
            print("    PASSED")
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Test 2: IterNode
        print("\n[2] IterNode...")
        try:
            nodes = list(IterNode("attn_out").iterate(n_layers=12, n_heads=12))
            assert len(nodes) == 12
            nodes2 = list(IterNode(["attn_out", "mlp_out"]).iterate(n_layers=12, n_heads=12))
            assert len(nodes2) == 24
            print(f"    Single component: {len(nodes)} nodes")
            print(f"    Two components: {len(nodes2)} nodes")
            print("    PASSED")
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Test 3: Load model and cache
        print("\n[3] cache_activations...")
        try:
            model = LanguageModel("gpt2", device_map="cpu")
            cache = cache_activations(model, ["Hello"], [
                Node("attn_out", 0), Node("mlp_out", 5), Node("resid_post", 11)
            ])
            assert len(cache) == 3
            for k, v in cache.items():
                print(f"    {k}: {v.shape}")
            print("    PASSED")
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Test 4: Activation patching
        print("\n[4] act_patch...")
        try:
            clean = ["When John and Mary went to the shops, Mary gave the bag to"]
            corrupt = ["When John and Mary went to the shops, John gave the bag to"]

            tok = model.tokenizer
            john_id = tok.encode(" John", add_special_tokens=False)[0]
            mary_id = tok.encode(" Mary", add_special_tokens=False)[0]

            def metric(logits):
                return (logits[:, -1, john_id] - logits[:, -1, mary_id]).mean()

            corrupt_cache = cache_activations(model, corrupt, [Node("attn_out", 9)])
            result = act_patch(model, clean, corrupt_cache, Node("attn_out", 9), metric)
            print(f"    Patched metric: {result.item():.4f}")
            print("    PASSED")
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Test 5: Path patching
        print("\n[5] path_patch...")
        try:
            result = path_patch(
                model, clean, corrupt,
                Node("attn_out", 9), Node("resid_post", 11), metric
            )
            print(f"    Path effect: {result.item():.4f}")
            print("    PASSED")
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Test 6: IterNode with path_patch
        print("\n[6] path_patch with IterNode...")
        try:
            results = path_patch(
                model, clean, corrupt,
                IterNode("attn_out"), Node("resid_post", 11), metric
            )
            print(f"    Shape: {results['attn_out'].shape}")
            print("    PASSED")
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        # Test 7: Causal ordering validation
        print("\n[7] Causal ordering check...")
        try:
            try:
                path_patch(model, clean, corrupt,
                          Node("resid_post", 11), Node("attn_out", 0), metric)
                print("    FAILED: Should have raised error")
                failed += 1
            except ValueError:
                print("    Correctly rejected invalid ordering")
                print("    PASSED")
                passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)
        return failed == 0

    success = run_tests()
    sys.exit(0 if success else 1)
