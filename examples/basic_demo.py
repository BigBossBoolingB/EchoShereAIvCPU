"""
Basic demonstration of EchoSphere AI-vCPU cognitive flow.

This demo shows the complete cognitive loop:
InputKSA → LogicKSA → MemoryKSA → OutputKSA

The demo analyzes chess pieces to demonstrate VSA similarity search
and Neo4j graph traversal working together.
"""

import time
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from echosphere.core import EchoSphere
from echosphere.utils.config import Config


def run_chess_analysis_demo():
    """
    Run the chess piece analysis demonstration.

    This demonstrates the complete cognitive flow:
    1. InputKSA receives "analyze Pawn" task
    2. LogicKSA processes task and queries MemoryKSA
    3. MemoryKSA finds similar concepts via VSA and relations via Neo4j
    4. OutputKSA generates coherent analysis result
    """
    print("=" * 60)
    print("EchoSphere AI-vCPU - Chess Analysis Demo")
    print("=" * 60)

    config = Config()
    config.log_level = "INFO"

    print("\n1. Starting EchoSphere system...")
    echosphere = EchoSphere(config)

    if not echosphere.start():
        print("❌ Failed to start EchoSphere system")
        return False

    print("✅ EchoSphere system started successfully")

    try:
        print("\n2. Waiting for system initialization...")
        time.sleep(2)

        stats = echosphere.get_system_stats()
        print(f"✅ System running: {stats['system']['is_running']}")
        print(f"✅ Workspace active: {stats['workspace']['active']}")

        print("\n3. Submitting analysis task: 'Pawn'")
        result = echosphere.submit_task("analyze", "Pawn")

        if result.get("status") == "success":
            print("✅ Task submitted successfully")
        else:
            print(f"❌ Task submission failed: {result}")
            return False

        print("\n4. Waiting for cognitive processing...")
        completed = echosphere.wait_for_completion(timeout=15.0)

        if completed:
            print("✅ Processing completed")
        else:
            print("⚠️  Processing timeout (may still be running)")

        print("\n5. Retrieving analysis results...")
        output_history = echosphere.get_output_history()

        if output_history:
            print("✅ Analysis results:")
            for i, output in enumerate(output_history, 1):
                print(f"\n--- Result {i} ---")
                print(f"Concept: {output['concept']}")
                print(f"Analysis: {output['analysis']}")
                print(f"Timestamp: {time.ctime(output['timestamp'])}")
        else:
            print("❌ No analysis results found")

        print("\n6. Memory system statistics:")
        memory_stats = echosphere.get_memory_stats()

        if "error" not in memory_stats:
            vsa_stats = memory_stats.get("vsa_memory", {})
            graph_stats = memory_stats.get("graph_store", {})
            query_stats = memory_stats.get("query_stats", {})

            print(
                f"VSA Memory: {vsa_stats.get('total_concepts', 0)} concepts, "
                f"{vsa_stats.get('dimensions', 0)} dimensions"
            )
            print(
                f"Graph Store: {graph_stats.get('concept_nodes', 0)} nodes, "
                f"{graph_stats.get('total_relationships', 0)} relationships"
            )
            print(
                f"Queries: {query_stats.get('total_queries', 0)} total, "
                f"{query_stats.get('success_rate', 0):.1f}% success rate"
            )
        else:
            print(f"❌ Memory stats error: {memory_stats['error']}")

        print("\n7. Testing additional concepts...")
        test_concepts = ["Queen", "Knight", "Rook"]

        for concept in test_concepts:
            print(f"\nAnalyzing: {concept}")
            result = echosphere.submit_task("analyze", concept)

            if result.get("status") == "success":
                print(f"✅ {concept} task submitted")
            else:
                print(f"❌ {concept} task failed: {result}")

        time.sleep(3)
        echosphere.wait_for_completion(timeout=10.0)

        print("\n8. Final analysis results:")
        final_outputs = echosphere.get_output_history()

        for output in final_outputs[-len(test_concepts) :]:
            print(f"• {output['concept']}: {output['analysis']}")

        print("\n" + "=" * 60)
        print("Demo completed successfully! 🎉")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logging.exception("Demo error details:")
        return False

    finally:
        print("\n9. Stopping EchoSphere system...")
        echosphere.stop()
        print("✅ System stopped")


def run_interactive_demo():
    """Run an interactive demo where users can submit their own analysis tasks."""
    print("=" * 60)
    print("EchoSphere AI-vCPU - Interactive Demo")
    print("=" * 60)

    config = Config()
    config.log_level = "WARNING"  # Reduce log noise for interactive mode

    echosphere = EchoSphere(config)

    if not echosphere.start():
        print("❌ Failed to start EchoSphere system")
        return

    print("✅ EchoSphere system started")
    print("\nYou can now submit analysis tasks!")
    print("Type 'quit' to exit, 'stats' for system statistics")
    print("Example: analyze Chess_Piece")

    try:
        while True:
            user_input = input("\nEnter task (analyze <concept>): ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "stats":
                stats = echosphere.get_system_stats()
                print(f"Tasks processed: {stats['system']['total_tasks_processed']}")
                print(f"Uptime: {stats['system']['uptime_seconds']:.1f} seconds")
                continue

            if user_input.startswith("analyze "):
                concept = user_input[8:].strip()
                if concept:
                    print(f"Analyzing: {concept}")
                    result = echosphere.submit_task("analyze", concept)

                    if result.get("status") == "success":
                        print("✅ Task submitted, processing...")

                        time.sleep(2)
                        outputs = echosphere.get_output_history()

                        if outputs:
                            latest = outputs[-1]
                            if latest["concept"] == concept:
                                print(f"Result: {latest['analysis']}")
                            else:
                                print("⏳ Still processing...")
                        else:
                            print("⏳ Still processing...")
                    else:
                        print(f"❌ Task failed: {result}")
                else:
                    print("❌ Please specify a concept to analyze")
            else:
                print("❌ Please use format: analyze <concept>")

    except KeyboardInterrupt:
        print("\n\nExiting...")

    finally:
        echosphere.stop()
        print("✅ System stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EchoSphere AI-vCPU Demo")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive demo"
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive_demo()
    else:
        success = run_chess_analysis_demo()
        sys.exit(0 if success else 1)
