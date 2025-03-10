import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from thoth.analyzer import ThothAnalyzer


def main() -> None:
    """
    Main entry point for the Thoth command-line tool

    Provides a command-line interface to use the ThothAnalyzer class
    """
    parser = argparse.ArgumentParser(description="Thoth - Semantic Analysis for Chat Data")

    # Basic parameters
    parser.add_argument("--db-path", type=str, help="Path to DuckDB database file")
    parser.add_argument("--phone", type=str, help="Phone number for user tables")
    parser.add_argument("--qdrant-path", type=str, default="./qdrant_db", help="Path to Qdrant database directory")
    parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Name of the Sentence Transformers model to use"
    )

    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import data from DuckDB")

    # Topic analysis commands
    topics_parser = subparsers.add_parser("topics", help="Analyze topics in chat data")
    topics_parser.add_argument(
        "--action",
        choices=["common", "evolution", "related", "keywords", "sentiment", "activity", "influence", "lifecycle"],
        help="Specific topic analysis to perform",
    )
    topics_parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters for topic modeling")
    topics_parser.add_argument("--time-window", type=str, default="1 month", help="Time window for analysis")
    topics_parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    topics_parser.add_argument("--top-n", type=int, default=10, help="Number of top items to return")
    topics_parser.add_argument("--topic-id", type=int, help="Specific topic ID to analyze")

    # User analysis commands
    users_parser = subparsers.add_parser("users", help="Analyze user behavior in chat data")
    users_parser.add_argument(
        "--action",
        choices=["topics", "interactions", "engagement", "behavior"],
        help="Specific user analysis to perform",
    )
    users_parser.add_argument("--min-messages", type=int, default=10, help="Minimum messages for user inclusion")
    users_parser.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions for user inclusion")

    # Cross-chat analysis
    cross_parser = subparsers.add_parser("cross", help="Cross-chat analysis")
    cross_parser.add_argument(
        "--action", choices=["topics", "correlations", "compare"], help="Specific cross-chat analysis to perform"
    )
    cross_parser.add_argument("--min-similarity", type=float, default=0.7, help="Minimum similarity threshold")
    cross_parser.add_argument("--min-correlation", type=float, default=0.5, help="Minimum correlation threshold")

    # Search functionality
    search_parser = subparsers.add_parser("search", help="Search chat data")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--chat-id", type=int, help="Specific chat ID to search")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum results to return")

    # Parse arguments
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ThothAnalyzer(
        db_path=args.db_path, phone=args.phone, embedding_model=args.embedding_model, qdrant_path=args.qdrant_path
    )

    # Process commands
    try:
        if args.command == "import":
            # Import data from DuckDB
            analyzer.import_from_duckdb()
            print("Data imported successfully")

        elif args.command == "topics":
            # Topic analysis
            if args.action == "common":
                result = analyzer.find_most_common_topic(n_clusters=args.n_clusters)
                print(f"Most common topic: {result[0]} (score: {result[1]:.2f})")

            elif args.action == "evolution":
                results = analyzer.find_topic_evolution(time_window=args.time_window)
                print(f"Topic evolution over time ({len(results)} time periods):")
                for i, period in enumerate(results):
                    print(f"\nPeriod {i+1}: {period['start_date']} to {period['end_date']}")
                    print(f"  Message count: {period['message_count']}")
                    print(f"  Dominant topic: {', '.join(period['dominant_topic'])}")

            elif args.action == "related":
                results = analyzer.find_related_topics(threshold=args.threshold)
                print(f"Related topics (threshold={args.threshold}):")
                for i, (topic1, topic2, score) in enumerate(results):
                    print(f"{i+1}. Topic {topic1} <-> Topic {topic2}: {score:.3f}")

            elif args.action == "keywords":
                results = analyzer.extract_topic_keywords(top_n=args.top_n)
                print(f"Topic keywords (top {args.top_n}):")
                for topic_id, keywords in results.items():
                    print(f"\nTopic {topic_id}:")
                    for keyword, score in keywords:
                        print(f"  {keyword}: {score:.3f}")

            elif args.action == "sentiment":
                if args.topic_id is not None:
                    results = analyzer.analyze_topic_sentiment(topic_id=args.topic_id)
                    print(f"Sentiment for Topic {args.topic_id}: {results.get(args.topic_id, 0):.3f}")
                else:
                    results = analyzer.analyze_topic_sentiment()
                    print("Topic sentiment scores:")
                    for topic_id, score in results.items():
                        print(f"  Topic {topic_id}: {score:.3f}")

            elif args.action == "activity":
                results = analyzer.analyze_topic_activity(time_window=args.time_window)
                print(f"Topic activity patterns (time window: {args.time_window}):")
                for topic_id, hours in results.items():
                    print(f"\nTopic {topic_id}:")
                    for hour, count in sorted(hours.items()):
                        if count > 0:
                            print(f"  {hour}: {count}")

            elif args.action == "influence":
                results = analyzer.analyze_topic_influence(min_replies=args.min_messages)
                print(f"Topic influence scores (min_replies={args.min_messages}):")
                for topic_id, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
                    print(f"  Topic {topic_id}: {score:.3f}")

            elif args.action == "lifecycle":
                results = analyzer.analyze_topic_lifecycle()
                print("Topic lifecycle analysis:")
                for topic_id, data in results.items():
                    print(f"\nTopic {topic_id} ({', '.join(data['keywords'])}):")
                    print(f"  Total messages: {data['total_messages']}")
                    print(f"  Lifecycle stage: {data['lifecycle_stage']}")
                    print(f"  Emergence date: {data['emergence_date']}")
                    print(f"  Peak date: {data['peak_date']}")
                    print(f"  Decline date: {data['decline_date']}")

            else:
                print("Please specify a topic analysis action")

        elif args.command == "users":
            # User analysis
            if args.action == "topics":
                results = analyzer.analyze_user_topics(min_messages=args.min_messages)
                print(f"User topic participation (min_messages={args.min_messages}):")
                for user_id, topics in results.items():
                    topic_counts = {}
                    for topic in topics:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1

                    print(f"\nUser {user_id}:")
                    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
                        print(f"  Topic {topic}: {count} messages")

            elif args.action == "interactions":
                results = analyzer.analyze_user_interactions(min_interactions=args.min_interactions)
                print(f"User interaction patterns (min_interactions={args.min_interactions}):")
                for i, interaction in enumerate(results):
                    print(
                        f"{i+1}. {interaction['from_user']} -> {interaction['to_user']}: {interaction['interaction_count']} interactions"
                    )

            elif args.action == "engagement":
                results = analyzer.analyze_user_engagement()
                print("User engagement metrics:")
                for user_id, metrics in results.items():
                    print(f"\nUser {user_id}:")
                    print(f"  Total messages: {metrics['total_messages']}")
                    print(f"  Overall reply ratio: {metrics['overall_reply_ratio']:.3f}")

                    if "topics" in metrics:
                        print("  Topic participation:")
                        for topic, data in metrics["topics"].items():
                            print(
                                f"    {topic}: {data['message_count']} messages, {data['reply_ratio']:.3f} reply ratio"
                            )

            elif args.action == "behavior":
                results = analyzer.analyze_user_behavior_patterns()
                print("User behavior patterns:")
                for user_id, pattern in results.items():
                    print(f"\nUser {user_id}:")
                    print(f"  Message count: {pattern['message_count']}")
                    print(f"  Reply ratio: {pattern['reply_ratio']:.3f}")
                    print(f"  Avg message length: {pattern['avg_message_length']:.1f} chars")

                    if pattern.get("avg_reply_time_seconds") is not None:
                        print(f"  Avg reply time: {pattern['avg_reply_time_seconds'] / 60:.1f} minutes")

                    print("  Most active hours:", end=" ")
                    hour_counts = pattern["hourly_pattern"]
                    top_hours = sorted(hour_counts.items(), key=lambda x: int(x[1]), reverse=True)[:3]
                    print(", ".join(f"{h}:00" for h, _ in top_hours))

            else:
                print("Please specify a user analysis action")

        elif args.command == "cross":
            # Cross-chat analysis
            if args.action == "topics":
                results = analyzer.analyze_cross_chat_topics(min_similarity=args.min_similarity)
                print(f"Cross-chat topics (min_similarity={args.min_similarity}):")
                for i, topic in enumerate(results):
                    print(f"\n{i+1}. Topic {topic['topic_id']} (keywords: {', '.join(topic['keywords'])}):")
                    print(
                        f"  Present in {len(topic['chat_distribution'])} chats with {topic['total_messages']} total messages"
                    )
                    for chat_id, count in topic["chat_distribution"].items():
                        print(f"    Chat {chat_id}: {count} messages")

            elif args.action == "correlations":
                results = analyzer.analyze_topic_correlations(min_correlation=args.min_correlation)
                print(f"Topic correlations (min_correlation={args.min_correlation}):")
                for i, corr in enumerate(results):
                    print(f"\n{i+1}. Topic {corr['topic_1']} <-> Topic {corr['topic_2']}: {corr['correlation']:.3f}")
                    print(f"  Topic {corr['topic_1']} keywords: {', '.join(corr['topic_1_keywords'])}")
                    print(f"  Topic {corr['topic_2']} keywords: {', '.join(corr['topic_2_keywords'])}")

            elif args.action == "compare":
                results = analyzer.compare_chats()
                print("Chat comparisons:")
                for i, comp in enumerate(results):
                    print(
                        f"\n{i+1}. Chat {comp['chat_id_1']} <-> Chat {comp['chat_id_2']}: {comp['similarity']:.3f} similarity"
                    )
                    print(f"  Topic similarity: {comp['topic_similarity']:.3f}")
                    print(f"  Activity pattern similarity: {comp['activity_pattern_similarity']:.3f}")
                    print(f"  Common active users: {comp['common_active_users']}")

            else:
                print("Please specify a cross-chat analysis action")

        elif args.command == "search":
            # Search functionality
            if args.chat_id is not None:
                results = analyzer.chat_search(args.chat_id, args.query, limit=args.limit)
                print(f"Search results for '{args.query}' in chat {args.chat_id}:")
            else:
                results = analyzer.search_messages(args.query, limit=args.limit)
                print(f"Search results for '{args.query}':")

            for i, result in enumerate(results):
                print(f"\n{i+1}. Score: {result['score']:.3f}")
                print(f"  Chat: {result.get('chat_id', 'unknown')}")
                print(f"  Date: {result.get('date', 'unknown')}")
                print(f"  From: {result.get('from_id', 'unknown')}")
                print(f"  Text: {result.get('text', '')[:100]}{'...' if len(result.get('text', '')) > 100 else ''}")

        else:
            print("Please specify a command. Use --help for available commands.")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    finally:
        # Clean up
        analyzer.close()


if __name__ == "__main__":
    main()
