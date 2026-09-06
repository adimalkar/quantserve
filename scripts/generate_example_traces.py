"""Generates realistic trace datasets for the three demonstration scenarios."""
import os
from quantserve.workload.synthetic import SyntheticTraceGenerator
from quantserve.workload.trace_parser import TraceParser


def main():
    os.makedirs("traces", exist_ok=True)
    os.makedirs("examples/coding_assistant", exist_ok=True)
    os.makedirs("examples/rag_assistant", exist_ok=True)
    os.makedirs("examples/batch_processing", exist_ok=True)

    # 1. Interactive Coding Assistant
    t_coding = SyntheticTraceGenerator.generate_interactive_coding(num_requests=250, seed=42)
    TraceParser.write_file(t_coding, "examples/coding_assistant/trace.jsonl")
    TraceParser.write_file(t_coding, "traces/coding_assistant.jsonl")
    print(f"Generated {len(t_coding)} records for Interactive Coding Assistant")

    # 2. RAG Knowledge Assistant
    t_rag = SyntheticTraceGenerator.generate_rag_assistant(num_requests=250, seed=1337)
    TraceParser.write_file(t_rag, "examples/rag_assistant/trace.jsonl")
    TraceParser.write_file(t_rag, "traces/rag_assistant.jsonl")
    print(f"Generated {len(t_rag)} records for RAG Knowledge Assistant")

    # 3. Batch Document Processing
    t_batch = SyntheticTraceGenerator.generate_batch_processing(num_requests=250, seed=999)
    TraceParser.write_file(t_batch, "examples/batch_processing/trace.jsonl")
    TraceParser.write_file(t_batch, "traces/batch_processing.jsonl")
    print(f"Generated {len(t_batch)} records for Batch Document Processing")


if __name__ == "__main__":
    main()
